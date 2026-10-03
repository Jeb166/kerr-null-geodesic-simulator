"""Sequential, headless same-definition spin sweep and batch solves.

Every spin builds its own stationary Kerr geometry; integrate_ray creates
new local frame, null covector and conserved quantities. Results retain
their own initial conditions. There is no UI, scheduler or approximation.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Callable, Iterable

import numpy as np

from .initial_conditions import RayDefinition
from .integration import (
    RELIABILITY_LIMIT, EventKind, IntegrationSettings, TerminationStatus,
    TrajectoryResult, integrate_ray,
)
from .kerr_schild import KerrSchildGeometry
from .parameters import SpacetimeParameters
from ._native import warm_compiled_rhs


class ReliabilityLevel(str, Enum):
    COMPLETE = "COMPLETE"       # certified finite event/fate classification
    PARTIAL = "PARTIAL"         # valid sampled prefix; no physical fate
    UNRELIABLE = "UNRELIABLE"   # solver failure, bad invariants or data


@dataclass(frozen=True, slots=True)
class ReliabilityAssessment:
    level: ReliabilityLevel
    physical_outcome: TerminationStatus | None
    reason: str


def assess_trajectory(result: TrajectoryResult) -> ReliabilityAssessment:
    """Apply the same numerical guard as integration and classify its status.

    ESCAPED is crossing a finite escape radius, and SINGULARITY_APPROACH
    means reaching a finite Sigma threshold, never a proven ring impact.
    MAX_INTEGRATION_LIMIT and CHART_BOUNDARY can have a valid sampled
    prefix but cannot supply a terminal physical outcome. A failed solver
    or the diagnostic gate invalidates physical fate claims.
    """
    if not isinstance(result, TrajectoryResult):
        raise TypeError("result must be a TrajectoryResult")
    values = (result.max_scaled_null_residual, result.max_scaled_Lz_drift,
              result.max_scaled_carter_constant_drift)
    if (len(result.affine_parameters) != len(result.states)
            or len(result.states) != len(result.sample_charts)
            or len(result.states) != len(result.sample_branches)
            or len(result.states) != len(result.sample_regions)
            or not np.all(np.isfinite(result.affine_parameters))
            or not np.all(np.isfinite(result.states))
            or not np.all(np.diff(result.affine_parameters) > 0)
            or not all(isfinite(v) for v in values)):
        return ReliabilityAssessment(ReliabilityLevel.UNRELIABLE, None,
                                     "nonfinite, disordered or inconsistent trajectory data")
    if (result.states.shape != (len(result.affine_parameters), 8)
            or not all(chart in ("ingoing", "outgoing") for chart in result.sample_charts)
            or not np.all(np.isin(result.sample_branches, (-1, 1)))
            or not all(isinstance(region, str) and region for region in result.sample_regions)
            or not isfinite(result.spin)
            or any(not isfinite(event.affine_parameter)
                   or event.affine_parameter < 0
                   or event.affine_parameter > result.affine_parameter_end
                   for event in result.region_transitions)):
        return ReliabilityAssessment(ReliabilityLevel.UNRELIABLE, None,
                                     "invalid chart, branch or event data")
    if (any(not isfinite(event.affine_parameter)
            or event.affine_parameter < 0
            or event.affine_parameter > result.affine_parameter_end
            for event in result.horizon_crossings)
            or (abs(result.spin) > 1 and result.horizon_crossings)
            or (abs(result.spin) in (0, 1) and
                any(event.kind == EventKind.INNER_HORIZON
                    for event in result.horizon_crossings))):
        return ReliabilityAssessment(ReliabilityLevel.UNRELIABLE, None,
                                     "horizon events contradict the Kerr spin regime")
    if any(v > RELIABILITY_LIMIT for v in values):
        return ReliabilityAssessment(ReliabilityLevel.UNRELIABLE, None,
                                     "scaled null/Lz/Carter reliability guard exceeded")
    if result.status in (TerminationStatus.NUMERICAL_ERROR,
                         TerminationStatus.DIAGNOSTIC_FAILURE):
        return ReliabilityAssessment(ReliabilityLevel.UNRELIABLE, None,
                                     f"integration status {result.status.value}")
    if result.status in (TerminationStatus.CHART_BOUNDARY,
                         TerminationStatus.MAX_INTEGRATION_LIMIT):
        return ReliabilityAssessment(ReliabilityLevel.PARTIAL, None,
                                     f"no physical terminal fate: {result.status.value}")
    if result.status in (TerminationStatus.ESCAPED,
                         TerminationStatus.SINGULARITY_APPROACH):
        try:
            final = KerrSchildGeometry(
                SpacetimeParameters(a_star=result.spin),
                branch=int(result.sample_branches[-1]),
                chart=result.sample_charts[-1],
            )
            info = final.domain(result.states[-1, 1:4])
            if result.status == TerminationStatus.ESCAPED:
                valid_endpoint = abs(abs(info.r) - result.settings.escape_radius) < 1e-7
            else:
                valid_endpoint = info.sigma is not None and info.sigma <= result.settings.epsilon_sigma + 1e-7
        except (ValueError, FloatingPointError, OverflowError):
            valid_endpoint = False
        if not valid_endpoint:
            return ReliabilityAssessment(ReliabilityLevel.UNRELIABLE, None,
                                         "recorded endpoint does not match its finite physical event")
        return ReliabilityAssessment(ReliabilityLevel.COMPLETE, result.status,
                                     f"reliable finite event: {result.status.value}")
    return ReliabilityAssessment(ReliabilityLevel.UNRELIABLE, None,
                                 "unrecognized termination status")


def _spin(a_star: Real) -> float:
    if isinstance(a_star, bool) or not isinstance(a_star, Real):
        raise ValueError("a_star must be a finite real number")
    try:
        converted = float(a_star)
    except (OverflowError, ValueError) as exc:
        raise ValueError("a_star must be a finite real number") from exc
    if not isfinite(converted):
        raise ValueError("a_star must be a finite real number")
    return converted


def solve_ray_for_spin(
    definition: RayDefinition, a_star: Real,
    settings: IntegrationSettings = IntegrationSettings(),
    *, backend: str = "auto", cancel_requested: Callable[[], bool] | None = None,
) -> TrajectoryResult:
    """Fresh geometry, observer and propagation for one saved ray definition."""
    if not isinstance(definition, RayDefinition):
        raise TypeError("definition must be a RayDefinition")
    geometry = KerrSchildGeometry(SpacetimeParameters(a_star=_spin(a_star)), backend=backend)
    return integrate_ray(geometry, definition, settings, cancel_requested=cancel_requested)


def sweep_ray_over_spins(
    definition: RayDefinition, spins: Iterable[Real],
    settings: IntegrationSettings = IntegrationSettings(),
    *, backend: str = "auto", cancel_requested: Callable[[], bool] | None = None,
) -> tuple[TrajectoryResult, ...]:
    """Sequential independent solves in input order, including repeated spins."""
    if not isinstance(definition, RayDefinition):
        raise TypeError("definition must be a RayDefinition")
    values = tuple(_spin(spin) for spin in spins)
    return tuple(solve_ray_for_spin(definition, spin, settings, backend=backend,
                                    cancel_requested=cancel_requested) for spin in values)


def solve_rays(
    definitions: Iterable[RayDefinition], a_star: Real,
    settings: IntegrationSettings = IntegrationSettings(),
    *, backend: str = "auto", cancel_requested: Callable[[], bool] | None = None,
) -> tuple[TrajectoryResult, ...]:
    """Solve a headless ray batch sequentially; reuse geometry, not momenta."""
    rays = tuple(definitions)
    if any(not isinstance(ray, RayDefinition) for ray in rays):
        raise TypeError("each definition must be a RayDefinition")
    geometry = KerrSchildGeometry(SpacetimeParameters(a_star=_spin(a_star)), backend=backend)
    return tuple(integrate_ray(geometry, ray, settings, cancel_requested=cancel_requested)
                 for ray in rays)


def warm_compute_backend() -> float | None:
    """Precompile the fused CPU kernel at startup; None means reference-only.

    Warmup is explicit and timed, so a first JIT compile is never folded
    into a claimed steady-state spin-update latency.
    """
    return warm_compiled_rhs()
