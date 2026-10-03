"""Latest-request-wins headless compute control, independent of physics/UI.

The coordinator owns one pending request at most. A newer generation drops
unstarted work and cooperatively interrupts currently running DOP853 solves.
Only fully computed results of the newest generation can be published.
"""

from dataclasses import dataclass
from enum import Enum
from threading import Condition, Event, Thread
from time import monotonic
from typing import Callable, Iterable

from .initial_conditions import RayDefinition
from .integration import ComputationCancelled, IntegrationSettings, TrajectoryResult
from .solving import _spin, solve_ray_for_spin, warm_compute_backend


class ComputeStatus(str, Enum):
    DONE = "DONE"
    SUPERSEDED = "SUPERSEDED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"  # compute infrastructure, not a geodesic fate


@dataclass(frozen=True, slots=True)
class RayEntry:
    ray_id: str
    definition: RayDefinition


@dataclass(frozen=True, slots=True)
class ComputeRequest:
    request_id: int
    spin: float
    rays: tuple[RayEntry, ...]
    settings: IntegrationSettings


@dataclass(frozen=True, slots=True)
class RayOutput:
    ray_id: str
    trajectory: TrajectoryResult


@dataclass(frozen=True, slots=True)
class ComputeSnapshot:
    request: ComputeRequest
    status: ComputeStatus
    rays: tuple[RayOutput, ...]
    elapsed_seconds: float
    message: str

    @property
    def publishable(self) -> bool:
        return self.status is ComputeStatus.DONE


def _freeze_trajectory(result: TrajectoryResult) -> None:
    """The coordinator alone owns these arrays; expose read-only snapshots."""
    for array in (
        result.states, result.affine_parameters, result.sample_branches,
        result.initial.state, result.initial.momentum, result.initial.tangent,
        result.initial.frame.observer, result.initial.frame.triad,
    ):
        array.setflags(write=False)
    for event in result.horizon_crossings:
        event.state.setflags(write=False)


class LatestWinsScheduler:
    """Warmed background CPU coordinator; no Qt, OpenGL or Kerr formulas.

    The default single coordinator can cancel between RHS callbacks. Its
    purpose is priority/stale-result safety; ``RayBatchExecutor`` independently
    supplies faster full-batch process parallelism. The two can be composed
    later without changing the result/publication contract.
    """

    def __init__(self, *, backend: str = "auto", solver: Callable = solve_ray_for_spin):
        if backend not in ("auto", "reference", "compiled"):
            raise ValueError("invalid backend")
        self.backend = backend
        self._solve = solver
        self._condition = Condition()
        self._thread: Thread | None = None
        self._closed = False
        self._latest_generation = 0
        self._pending: ComputeRequest | None = None
        self._active_token: Event | None = None
        self._latest_snapshot: ComputeSnapshot | None = None
        self._outcomes: dict[int, ComputeSnapshot] = {}
        self.warmup_seconds: float | None = None

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("scheduler is closed")
        if self._thread is not None:
            return
        self.warmup_seconds = warm_compute_backend() if self.backend != "reference" else None
        if self.backend == "compiled" and self.warmup_seconds is None:
            raise RuntimeError("compiled scheduler requires Numba")
        self._thread = Thread(target=self._run, name="headless-ray-coordinator", daemon=True)
        self._thread.start()

    def submit(
        self, spin: float, rays: Iterable[RayDefinition | RayEntry],
        settings: IntegrationSettings = IntegrationSettings(),
    ) -> ComputeRequest:
        if self._thread is None or self._closed:
            raise RuntimeError("start the scheduler before submitting")
        value = _spin(spin)
        if not isinstance(settings, IntegrationSettings):
            raise TypeError("settings must be IntegrationSettings")
        supplied = tuple(rays)
        entries = []
        for index, item in enumerate(supplied):
            if isinstance(item, RayEntry):
                ray_id, source = item.ray_id, item.definition
            elif isinstance(item, RayDefinition):
                ray_id, source = f"ray-{index:04d}", item
            else:
                raise TypeError("rays must be RayDefinition or RayEntry objects")
            if not isinstance(ray_id, str) or not ray_id or not isinstance(source, RayDefinition):
                raise ValueError("each ray needs a nonempty ID and RayDefinition")
            frozen_copy = RayDefinition(
                source.coordinate_position, source.local_direction,
                coordinate_time=source.coordinate_time,
                omega_local=source.omega_local,
                frame_convention=source.frame_convention,
            )
            entries.append(RayEntry(ray_id, frozen_copy))
        if len({item.ray_id for item in entries}) != len(entries):
            raise ValueError("ray IDs must be unique within a request")
        with self._condition:
            if self._closed:
                raise RuntimeError("scheduler is closed")
            self._latest_generation += 1
            request = ComputeRequest(self._latest_generation, value, tuple(entries), settings)
            if self._pending is not None:
                self._publish(ComputeSnapshot(
                    self._pending, ComputeStatus.SUPERSEDED, (), 0.,
                    "unstarted request discarded for newer spin",
                ))
            if self._active_token is not None:
                self._active_token.set()
            self._pending = request
            self._latest_snapshot = None
            self._condition.notify_all()
            return request

    def _publish(self, snapshot: ComputeSnapshot) -> None:
        """Called under the coordinator condition lock."""
        self._outcomes[snapshot.request.request_id] = snapshot
        if len(self._outcomes) > 256:
            del self._outcomes[next(iter(self._outcomes))]
        if snapshot.publishable and snapshot.request.request_id == self._latest_generation:
            self._latest_snapshot = snapshot
        self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closed)
                if self._closed:
                    return
                request = self._pending
                self._pending = None
                token = Event()
                self._active_token = token
            started = monotonic()
            completed = []
            try:
                for item in request.rays:
                    if token.is_set():
                        raise ComputationCancelled("newer generation superseded this batch")
                    solved = self._solve(item.definition, request.spin, request.settings,
                                         backend=self.backend, cancel_requested=token.is_set)
                    if token.is_set():
                        raise ComputationCancelled("newer generation superseded this batch")
                    _freeze_trajectory(solved)
                    completed.append(RayOutput(item.ray_id, solved))
                status, message = ComputeStatus.DONE, "all requested rays solved"
            except ComputationCancelled as exc:
                status, message = ComputeStatus.CANCELLED, str(exc)
                completed = []
            except Exception as exc:
                status, message = ComputeStatus.FAILED, f"{type(exc).__name__}: {exc}"
                completed = []
            with self._condition:
                self._active_token = None
                if request.request_id != self._latest_generation:
                    status, message = ComputeStatus.SUPERSEDED, "stale generation, never published"
                    completed = []
                elif self._closed:
                    status, message = ComputeStatus.CANCELLED, "scheduler closed"
                    completed = []
                self._publish(ComputeSnapshot(request, status, tuple(completed),
                                              monotonic()-started, message))

    def latest(self) -> ComputeSnapshot | None:
        with self._condition:
            return self._latest_snapshot

    def wait_for(self, request_id: int, timeout: float | None = None) -> ComputeSnapshot:
        with self._condition:
            if not self._condition.wait_for(lambda: request_id in self._outcomes, timeout=timeout):
                raise TimeoutError(f"request {request_id} did not finish in time")
            return self._outcomes[request_id]

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            if self._pending is not None:
                self._publish(ComputeSnapshot(
                    self._pending, ComputeStatus.CANCELLED, (), 0., "scheduler closed before start"))
                self._pending = None
            if self._active_token is not None:
                self._active_token.set()
            self._latest_snapshot = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10.)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
