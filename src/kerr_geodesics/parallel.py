"""Persistent headless ray-level CPU workers; no physics or UI logic."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from numbers import Integral
from typing import Iterable

from ._native import HAS_NUMBA
from .initial_conditions import RayDefinition
from .integration import IntegrationSettings, TrajectoryResult
from .solving import _spin, solve_ray_for_spin, warm_compute_backend


def _worker_warm():
    warm_compute_backend()


def _worker_ping():
    # Used during explicit startup, never timed as a warmed batch solve.
    return True


def _solve_one(payload):
    index, definition, spin, settings, backend = payload
    # Re-apply the immutable-definition copy on a process-pickle boundary.
    copied = RayDefinition(
        definition.coordinate_position, definition.local_direction,
        coordinate_time=definition.coordinate_time,
        omega_local=definition.omega_local,
        frame_convention=definition.frame_convention,
    )
    return index, solve_ray_for_spin(copied, spin, settings, backend=backend)


class RayBatchExecutor:
    """Reusable worker pool; ordered full-accuracy solves from isolated rays.

    ``start()`` pays worker spawn and kernel warmup before live requests.
    The Python reference path and unpooled sequential ``solve_rays`` remain
    independent. Closing the context waits for all submitted tasks.
    """

    def __init__(self, workers: int = 4, *, backend: str = "auto"):
        if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if backend not in ("auto", "reference", "compiled"):
            raise ValueError("invalid physics backend")
        if backend == "compiled" and not HAS_NUMBA:
            raise RuntimeError("compiled backend requires the Numba performance extra")
        self.workers = int(workers)
        self.backend = backend
        self._pool = None

    def start(self) -> None:
        if self._pool is not None:
            return
        self._pool = ProcessPoolExecutor(
            max_workers=self.workers, mp_context=get_context("spawn"),
            initializer=_worker_warm,
        )
        # Force all processes to initialize and load the kernel before timing.
        tasks = [self._pool.submit(_worker_ping) for _ in range(self.workers)]
        for task in tasks:
            task.result()

    def solve_rays(
        self, definitions: Iterable[RayDefinition], a_star: float,
        settings: IntegrationSettings = IntegrationSettings(),
    ) -> tuple[TrajectoryResult, ...]:
        rays = tuple(definitions)
        if any(not isinstance(ray, RayDefinition) for ray in rays):
            raise TypeError("each definition must be a RayDefinition")
        spin = _spin(a_star)
        if self._pool is None:
            raise RuntimeError("call start() before a warmed parallel solve")
        tasks = [self._pool.submit(_solve_one, (i, ray, spin, settings, self.backend))
                 for i, ray in enumerate(rays)]
        try:
            indexed = (task.result() for task in tasks)
            return tuple(result for _, result in sorted(indexed, key=lambda item: item[0]))
        except BaseException:
            for task in tasks:
                task.cancel()
            raise

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
