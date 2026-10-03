"""One-ray, forward-affine DOP853 integration with explicit geometric events.

The ODE has seven evolved variables (t,x,y,z,p_x,p_y,p_z); p_t is restored
from the null initial condition and held exactly fixed by stationarity.
No metric, radial-gradient, horizon or Carter formula is defined here.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import DOP853
from scipy.optimize import brentq

from .geometry import Geometry
from .hamiltonian import null_geodesic_rhs
from .initial_conditions import InitialConditions, RayDefinition, create_initial_conditions


# Empirical constraint guard introduced in v0.1E. This is not a proven global
# integration error bound; it is fixed across production solves and validation.
RELIABILITY_LIMIT = 1e-8


class ComputationCancelled(Exception):
    """Compute-control signal, never a geodesic or physics terminal status."""


class TerminationStatus(str, Enum):
    ESCAPED = "ESCAPED"
    SINGULARITY_APPROACH = "SINGULARITY_APPROACH"
    CHART_BOUNDARY = "CHART_BOUNDARY"
    MAX_INTEGRATION_LIMIT = "MAX_INTEGRATION_LIMIT"
    NUMERICAL_ERROR = "NUMERICAL_ERROR"
    DIAGNOSTIC_FAILURE = "DIAGNOSTIC_FAILURE"


class EventKind(str, Enum):
    OUTER_HORIZON = "OUTER_HORIZON"
    INNER_HORIZON = "INNER_HORIZON"


@dataclass(frozen=True, slots=True)
class IntegrationSettings:
    """M=1 units; escape_radius is a finite coordinate boundary, not infinity."""

    escape_radius: float = 12.0
    max_affine_parameter: float = 60.0
    max_accepted_steps: int = 10000
    rtol: float = 1e-10
    # (t,x,y,z,p_x,p_y,p_z); near-zero spatial momenta require absolute control.
    atol: tuple[float, ...] = (1e-11, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12)
    max_step: float = 0.5
    epsilon_sigma: float = 1e-4
    chart_boundary_radius: float = 1e-3
    approach_step_fraction: float = 0.15

    def __post_init__(self):
        positives = (
            self.escape_radius, self.max_affine_parameter, self.rtol,
            self.max_step, self.epsilon_sigma, self.chart_boundary_radius,
            self.approach_step_fraction,
        )
        if any(not isfinite(v) or v <= 0.0 for v in positives):
            raise ValueError("integration settings must be finite and positive")
        if isinstance(self.max_accepted_steps, bool) or not isinstance(self.max_accepted_steps, int) or self.max_accepted_steps < 1:
            raise ValueError("max_accepted_steps must be a positive integer")
        if len(self.atol) != 7 or any(not isfinite(v) or v <= 0 for v in self.atol):
            raise ValueError("atol must contain seven finite, positive component tolerances")
        if self.approach_step_fraction >= 1.0:
            raise ValueError("approach_step_fraction must be below 1")


@dataclass(frozen=True, slots=True)
class HorizonCrossing:
    kind: EventKind
    affine_parameter: float
    direction: str  # INWARD or OUTWARD, determined by the sign change in r
    state: NDArray[np.float64]
    radial_coordinate: float
    chart: str = "ingoing"
    branch: int = 1


@dataclass(frozen=True, slots=True)
class RegionTransition:
    kind: str  # OUTER_HORIZON, INNER_HORIZON, DISK_CROSSING, CHART_HANDOFF
    affine_parameter: float
    from_region: str
    to_region: str
    from_chart: str
    to_chart: str


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    status: TerminationStatus
    message: str
    affine_parameters: NDArray[np.float64]
    states: NDArray[np.float64]  # samples in full 8-component convention
    horizon_crossings: tuple[HorizonCrossing, ...]
    initial: InitialConditions
    settings: IntegrationSettings
    spin: float
    accepted_steps: int
    rhs_evaluations: int
    rejected_steps: int | None  # SciPy's public solver interface does not expose this.
    minimum_accepted_step: float | None
    minimum_sampled_r: float
    minimum_sampled_sigma: float
    max_abs_null_residual: float  # abs(g^mu nu p_mu p_nu), at recorded states/events
    max_scaled_null_residual: float
    max_abs_energy_drift: float
    max_abs_Lz_drift: float
    max_scaled_Lz_drift: float
    max_abs_carter_constant_drift: float
    max_scaled_carter_constant_drift: float
    minimum_sampled_abs_r: float
    sample_branches: NDArray[np.int8]
    sample_charts: tuple[str, ...]
    sample_regions: tuple[str, ...]
    region_transitions: tuple[RegionTransition, ...]

    @property
    def affine_parameter_end(self) -> float:
        return float(self.affine_parameters[-1])


def integrate_ray(
    geometry: Geometry, definition: RayDefinition,
    settings: IntegrationSettings = IntegrationSettings(),
    *, cancel_requested: Callable[[], bool] | None = None,
) -> TrajectoryResult:
    """Integrate one null ray up to a finite event or compute budget.

    A DOP853 accepted step gets a dense interpolant. Bracketed zeros of
    geometric event functions are refined with Brent's method; a horizon
    event records its root and continues. Finite-difference derivatives,
    post-step null projection and coordinate-time integration are absent.
    """
    if cancel_requested is not None and cancel_requested():
        raise ComputationCancelled("superseded before ray initialization")
    initial = create_initial_conditions(geometry, definition)
    active_geometry = geometry
    initial_p_t = float(initial.momentum[0])
    initial_conserved = initial.conserved
    times: list[float] = [0.0]
    states: list[NDArray[np.float64]] = [initial.state.copy()]
    crossings: list[HorizonCrossing] = []
    transitions: list[RegionTransition] = []
    branches: list[int] = []
    charts: list[str] = []
    regions: list[str] = []
    accepted_steps = 0
    minimum_step: float | None = None
    max_null = max_scaled_null = max_E = max_Lz = max_scaled_Lz = 0.0
    max_carter = max_scaled_carter = 0.0
    minimum_r = minimum_abs_r = minimum_sigma = float("inf")
    solver: DOP853 | None = None
    nfev_previous_segments = 0
    chart_switch_radius: float | None = None

    def region(position: NDArray[np.float64]) -> str:
        radius = active_geometry.radial_coordinate(position)
        horizon = active_geometry.horizons()
        if radius < 0:
            return "NEGATIVE_R"
        if radius == 0:
            return "REGULAR_DISK"
        if not horizon.exists:
            return "SUPER_EXTREMAL_POSITIVE" if active_geometry.parameters.a_star**2 > 1 else "POSITIVE_R"
        if horizon.has_distinct_inner_horizon and radius < horizon.r_minus:
            return "INNER"
        if radius < horizon.r_plus:
            return ("EXTREMAL_INTERIOR" if horizon.r_plus == horizon.r_minus
                    else "BETWEEN_HORIZONS")
        return "EXTERIOR"

    def swap(new_geometry: Geometry) -> None:
        nonlocal active_geometry
        active_geometry = new_geometry
        if solver is not None and solver.status == "running":
            # DOP853 caches f(t,y) across steps; the fields agree at the
            # switch but explicitly refresh to avoid a stale chart tangent.
            solver.f = rhs(solver.t, solver.y)

    def full_state(y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.r_[y[:4], initial_p_t, y[4:7]]

    def rhs(affine: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        if cancel_requested is not None and cancel_requested():
            raise ComputationCancelled("superseded during DOP853 evaluation")
        full = full_state(y)
        evaluated = active_geometry.fast_hamiltonian_rhs(full)
        if evaluated is None:
            evaluated = null_geodesic_rhs(active_geometry, full)
        return np.r_[evaluated[:4], evaluated[5:8]]

    def record(full: NDArray[np.float64]) -> None:
        nonlocal minimum_r, minimum_abs_r, minimum_sigma, max_null, max_scaled_null
        nonlocal max_E, max_Lz, max_scaled_Lz, max_carter, max_scaled_carter
        info = active_geometry.domain(full[1:4])
        minimum_r = min(minimum_r, info.r)
        minimum_abs_r = min(minimum_abs_r, abs(info.r))
        minimum_sigma = min(minimum_sigma, info.sigma)
        p = full[4:8]
        inverse = active_geometry.inverse_metric(full[1:4])
        residual = abs(float(p @ inverse @ p))
        residual_scale = max(1.0, float(np.linalg.norm(inverse, ord=np.inf) * (p @ p)))
        max_null = max(max_null, residual)
        max_scaled_null = max(max_scaled_null, residual / residual_scale)
        quantities = active_geometry.conserved_quantities(full[1:4], p)
        dE = abs(quantities.energy - initial_conserved.energy)
        dL = abs(quantities.angular_momentum_z - initial_conserved.angular_momentum_z)
        dC = abs(quantities.carter_constant - initial_conserved.carter_constant)
        max_E = max(max_E, dE)
        max_Lz = max(max_Lz, dL)
        max_scaled_Lz = max(max_scaled_Lz, dL / max(1.0, abs(initial_conserved.angular_momentum_z)))
        max_carter = max(max_carter, dC)
        max_scaled_carter = max(
            max_scaled_carter,
            dC / max(1.0, abs(initial_conserved.carter_constant), initial_conserved.energy**2),
        )

    def append_sample(affine: float, full: NDArray[np.float64]) -> None:
        times.append(affine)
        states.append(full)
        record(full)
        radius = active_geometry.radial_coordinate(full[1:4])
        branches.append(1 if radius >= 0 else -1)
        charts.append(getattr(active_geometry, "chart", "unspecified"))
        regions.append(region(full[1:4]))

    def diagnostic_error() -> str | None:
        if (max_scaled_null > RELIABILITY_LIMIT or max_scaled_Lz > RELIABILITY_LIMIT
                or max_scaled_carter > RELIABILITY_LIMIT):
            return ("null/Lz/Carter reliability gate exceeded: "
                    f"{max_scaled_null:.3e}/{max_scaled_Lz:.3e}/{max_scaled_carter:.3e}")
        return None

    def finish(status: TerminationStatus, message: str) -> TrajectoryResult:
        return TrajectoryResult(
            status=status, message=message,
            affine_parameters=np.array(times, dtype=np.float64),
            states=np.array(states, dtype=np.float64),
            horizon_crossings=tuple(crossings), initial=initial, settings=settings,
            spin=float(geometry.parameters.a_star),
            accepted_steps=accepted_steps,
            rhs_evaluations=nfev_previous_segments+(solver.nfev if solver is not None else 0),
            rejected_steps=None, minimum_accepted_step=minimum_step,
            minimum_sampled_r=minimum_r, minimum_sampled_sigma=minimum_sigma,
            max_abs_null_residual=max_null, max_scaled_null_residual=max_scaled_null,
            max_abs_energy_drift=max_E, max_abs_Lz_drift=max_Lz,
            max_scaled_Lz_drift=max_scaled_Lz,
            max_abs_carter_constant_drift=max_carter,
            max_scaled_carter_constant_drift=max_scaled_carter,
            minimum_sampled_abs_r=minimum_abs_r,
            sample_branches=np.array(branches, dtype=np.int8),
            sample_charts=tuple(charts), sample_regions=tuple(regions),
            region_transitions=tuple(sorted(transitions, key=lambda event: event.affine_parameter)),
        )

    def measure(label: str, y: NDArray[np.float64]) -> float:
        position = y[1:4]
        if label == "sigma":
            return active_geometry.domain(position).sigma - settings.epsilon_sigma
        if label == "chart":
            distance = active_geometry.chart_boundary_coordinate(position)
            return 1.0 if distance is None else distance - settings.chart_boundary_radius
        radial = active_geometry.radial_coordinate(position)
        if label == "escape":
            return abs(radial) - settings.escape_radius
        if label == "outer":
            return radial - active_geometry.horizons().r_plus
        if label == "inner":
            return radial - active_geometry.horizons().r_minus
        if label == "disk":
            return float(position[2])
        if label == "turn":
            return radial_speed(full_state(y))
        if label == "preferred_chart":
            return radial - chart_switch_radius
        raise AssertionError(label)

    def radial_speed(full: NDArray[np.float64]) -> float:
        evaluated = active_geometry.fast_hamiltonian_rhs(full)
        if evaluated is None:
            evaluated = null_geodesic_rhs(active_geometry, full)
        velocity = evaluated[1:4]
        return float(active_geometry.radial_coordinate_gradient(full[1:4]) @ velocity)

    def crossed(label: str, left: float, right: float) -> bool:
        if label == "escape":
            return left < 0.0 <= right
        if label == "turn":
            return (left < 0.0 <= right) or (left > 0.0 >= right)
        if label in ("sigma", "chart", "preferred_chart"):
            return left > 0.0 >= right
        return (left > 0.0 >= right) or (left < 0.0 <= right)

    try:
        record(initial.state)
        branches.append(getattr(active_geometry, "branch", 1))
        charts.append(getattr(active_geometry, "chart", "unspecified"))
        regions.append(region(initial.state[1:4]))
        if measure("sigma", initial.state[:4]) <= 0:
            return finish(TerminationStatus.SINGULARITY_APPROACH, "initial Sigma already at the configured approach threshold")
        if measure("chart", initial.state[:4]) <= 0:
            return finish(TerminationStatus.CHART_BOUNDARY, "initial point already at a chart guard")
        y0 = np.r_[initial.state[:4], initial.state[5:8]]
        solver = DOP853(
            rhs, t0=0.0, y0=y0, t_bound=settings.max_affine_parameter,
            rtol=settings.rtol, atol=np.asarray(settings.atol),
            max_step=settings.max_step,
        )
        horizon = active_geometry.horizons()
        labels = ["escape", "sigma"]
        if active_geometry.chart_boundary_coordinate(initial.state[1:4]) is not None:
            labels.append("chart")
        if horizon.exists:
            labels.append("outer")
            if horizon.has_distinct_inner_horizon:
                labels.append("inner")

        while solver.status == "running":
            if cancel_requested is not None and cancel_requested():
                raise ComputationCancelled("superseded between accepted steps")
            if accepted_steps >= settings.max_accepted_steps:
                return finish(TerminationStatus.MAX_INTEGRATION_LIMIT, "accepted-step budget exhausted")

            current = full_state(solver.y)
            extension = active_geometry.disk_extension(current[1:4])
            if extension is not None:
                swap(extension)
            info = active_geometry.domain(current[1:4])
            solver.max_step = settings.max_step
            if getattr(active_geometry, "disk_hemisphere", None) is not None and horizon.exists and active_geometry.parameters.a_star != 0:
                speed = abs(radial_speed(current))
                solver.max_step = min(solver.max_step,
                                      .25*horizon.r_minus/max(speed, 1e-14))
            if (abs(info.r) < 1.5 or info.sigma < 1.0) and getattr(active_geometry, "disk_hemisphere", None) is None:
                speed = abs(radial_speed(current))
                if speed > 1e-14:
                    solver.max_step = min(
                        settings.max_step,
                        settings.approach_step_fraction * abs(info.r) / speed,
                    )
            left_time, left_y = float(solver.t), solver.y.copy()
            step_message = solver.step()
            if solver.status == "failed":
                return finish(TerminationStatus.NUMERICAL_ERROR, f"DOP853 failed: {step_message}")
            accepted_steps += 1
            right_time, right_y = float(solver.t), solver.y.copy()
            step_size = right_time - left_time
            minimum_step = step_size if minimum_step is None else min(minimum_step, step_size)
            dense = solver.dense_output()

            roots: list[tuple[float, str, NDArray[np.float64], str]] = []
            dynamic_labels = labels.copy()
            preferred_chart = active_geometry.chart_transition_surface(full_state(left_y))
            chart_switch_radius = preferred_chart[1] if preferred_chart is not None else None
            if preferred_chart is not None:
                dynamic_labels.append("preferred_chart")
            if getattr(active_geometry, "disk_hemisphere", None) is not None:
                dynamic_labels.append("disk")
            left_radius = active_geometry.radial_coordinate(left_y[1:4])
            radial_turn_limit = horizon.r_minus if horizon.exists else 1.0
            turn_inward = (active_geometry.parameters.a_star != 0
                           and getattr(active_geometry, "chart", None) == "ingoing"
                           and 0 < left_radius < radial_turn_limit)
            turn_outward = (getattr(active_geometry, "chart", None) == "outgoing"
                            and left_radius > (horizon.r_plus if horizon.exists else 1.0))
            if turn_inward or turn_outward:
                dynamic_labels.append("turn")
            for label in dynamic_labels:
                left_value = measure(label, left_y)
                right_value = measure(label, right_y)
                if not crossed(label, left_value, right_value):
                    continue
                if label == "turn" and not ((turn_inward and left_value < 0 <= right_value)
                                             or (turn_outward and left_value > 0 >= right_value)):
                    continue
                root_time = (
                    right_time if right_value == 0.0 else
                    float(brentq(
                        lambda lam: measure(label, dense(lam)),
                        left_time, right_time, xtol=1e-12,
                    ))
                )
                root_state = full_state(dense(root_time))
                direction = "OUTWARD" if right_value > left_value else "INWARD"
                if label == "escape" and np.sign(active_geometry.radial_coordinate(root_state[1:4]))*radial_speed(root_state) <= 0.0:
                    continue
                if label == "disk" and np.hypot(*root_state[1:3]) >= abs(active_geometry.parameters.a_star):
                    continue
                roots.append((root_time, label, root_state, direction))

            terminal = min(
                (item for item in roots if item[1] in ("escape", "sigma", "chart", "turn", "preferred_chart")),
                key=lambda item: item[0], default=None,
            )
            terminal_time = terminal[0] if terminal is not None else right_time
            for root_time, label, full, direction in sorted(roots, key=lambda item: item[0]):
                if label in ("outer", "inner") and root_time <= terminal_time:
                    crossings.append(HorizonCrossing(
                        kind=EventKind.OUTER_HORIZON if label == "outer" else EventKind.INNER_HORIZON,
                        affine_parameter=root_time, direction=direction,
                        state=full, radial_coordinate=active_geometry.radial_coordinate(full[1:4]),
                        chart=getattr(active_geometry, "chart", "unspecified"),
                        branch=getattr(active_geometry, "branch", 1),
                    ))
                    record(full)
                    inner_band = ("EXTREMAL_INTERIOR" if horizon.r_plus == horizon.r_minus
                                  else "BETWEEN_HORIZONS")
                    transitions.append(RegionTransition(
                        label.upper()+"_HORIZON", root_time,
                        "EXTERIOR" if label == "outer" and direction == "INWARD" else
                        "INNER" if label == "inner" and direction == "OUTWARD" else
                        inner_band,
                        inner_band if label == "outer" and direction == "INWARD" else
                        "BETWEEN_HORIZONS" if label == "inner" and direction == "OUTWARD" else
                        "INNER" if label == "inner" else "EXTERIOR",
                        getattr(active_geometry, "chart", "unspecified"),
                        getattr(active_geometry, "chart", "unspecified"),
                    ))

            for root_time, label, full, direction in sorted(roots, key=lambda item: item[0]):
                if label == "disk" and root_time <= terminal_time:
                    from_positive = left_radius > 0
                    transitions.append(RegionTransition(
                        "DISK_CROSSING", root_time,
                        "POSITIVE_R" if from_positive else "NEGATIVE_R",
                        "NEGATIVE_R" if from_positive else "POSITIVE_R",
                        getattr(active_geometry, "chart", "unspecified"),
                        getattr(active_geometry, "chart", "unspecified"),
                    ))
                    if root_time > times[-1]:
                        append_sample(root_time, full)

            if terminal is not None:
                root_time, label, root_state, _ = terminal
                if label in ("turn", "preferred_chart"):
                    target_chart = (preferred_chart[0] if label == "preferred_chart"
                                    else "outgoing" if turn_inward else "ingoing")
                    handoff = active_geometry.horizon_chart_handoff(root_state, target_chart)
                    if handoff is None:
                        return finish(TerminationStatus.CHART_BOUNDARY, "horizon-penetrating patch handoff unavailable")
                    before = region(root_state[1:4])
                    old_chart = getattr(active_geometry, "chart", "unspecified")
                    new_geometry, switched = handoff
                    nfev_previous_segments += solver.nfev
                    active_geometry = new_geometry
                    if abs(switched[4]-initial_p_t) > 1e-9*max(1.,abs(initial_p_t)):
                        return finish(TerminationStatus.DIAGNOSTIC_FAILURE, "chart handoff changed Killing energy")
                    switched[4] = initial_p_t
                    transitions.append(RegionTransition(
                        "CHART_HANDOFF", root_time, before, region(switched[1:4]),
                        old_chart, getattr(active_geometry, "chart", "unspecified"),
                    ))
                    if root_time > times[-1]:
                        append_sample(root_time, switched)
                    solver = DOP853(rhs, t0=root_time,
                                    y0=np.r_[switched[:4], switched[5:8]],
                                    t_bound=settings.max_affine_parameter,
                                    rtol=settings.rtol, atol=np.asarray(settings.atol),
                                    max_step=settings.max_step)
                    continue
                if root_time > times[-1]:
                    append_sample(root_time, root_state)
                status = {
                    "escape": TerminationStatus.ESCAPED,
                    "sigma": TerminationStatus.SINGULARITY_APPROACH,
                    "chart": TerminationStatus.CHART_BOUNDARY,
                }[label]
                if diagnostic_error():
                    return finish(TerminationStatus.DIAGNOSTIC_FAILURE, diagnostic_error())
                return finish(status, f"crossed {label} finite threshold (root localized with dense output)")

            full_right = full_state(right_y)
            append_sample(right_time, full_right)
            if diagnostic_error():
                return finish(TerminationStatus.DIAGNOSTIC_FAILURE, diagnostic_error())
            fixed = active_geometry.fixed_branch(full_right[1:4])
            if fixed is not None:
                swap(fixed)
                if (getattr(active_geometry, "chart", None) == "ingoing"
                        and horizon.exists and active_geometry.parameters.a_star != 0
                        and active_geometry.radial_coordinate(full_right[1:4]) > 0
                        and solver.status == "running"):
                    handoff = active_geometry.horizon_chart_handoff(full_right, "outgoing")
                    if handoff is None:
                        return finish(TerminationStatus.CHART_BOUNDARY, "disk exit needs outgoing horizon patch")
                    new_geometry, switched = handoff
                    nfev_previous_segments += solver.nfev
                    active_geometry = new_geometry
                    if abs(switched[4]-initial_p_t) > 1e-9*max(1.,abs(initial_p_t)):
                        return finish(TerminationStatus.DIAGNOSTIC_FAILURE, "disk chart handoff changed Killing energy")
                    switched[4] = initial_p_t
                    transitions.append(RegionTransition(
                        "CHART_HANDOFF", right_time, region(full_right[1:4]),
                        region(switched[1:4]), "ingoing", "outgoing"))
                    states[-1] = switched
                    branches[-1] = 1
                    charts[-1] = "outgoing"
                    regions[-1] = region(switched[1:4])
                    record(switched)
                    solver = DOP853(rhs, t0=right_time,
                                    y0=np.r_[switched[:4], switched[5:8]],
                                    t_bound=settings.max_affine_parameter,
                                    rtol=settings.rtol, atol=np.asarray(settings.atol),
                                    max_step=settings.max_step)
                    continue

        return finish(TerminationStatus.MAX_INTEGRATION_LIMIT, "affine-parameter budget exhausted")
    except ComputationCancelled:
        raise
    except Exception as exc:
        return finish(TerminationStatus.NUMERICAL_ERROR, f"{type(exc).__name__}: {exc}")
