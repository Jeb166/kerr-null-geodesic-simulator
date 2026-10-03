"""Immutable rendering snapshots and CPU-only display mesh conversion.

All chart-specific coordinate transforms are delegated to physics geometry.
These functions never modify integration samples or feed GPU data back to it.
"""

from dataclasses import dataclass
from math import cos, pi, sin, sqrt

import numpy as np
from numpy.typing import NDArray

from kerr_geodesics import (
    HorizonInfo, HorizonRegime, KerrSchildGeometry,
    SingularityInfo, SpacetimeParameters, TrajectoryResult, assess_trajectory,
)


def _readonly(values, *, dtype=np.float64):
    owned = np.array(values, dtype=dtype, copy=True, order="C")
    # A view of immutable bytes cannot be made writable with setflags(True).
    return np.frombuffer(owned.tobytes(), dtype=dtype).reshape(owned.shape)


@dataclass(frozen=True, slots=True)
class DisplayRay:
    name: str
    positions: NDArray[np.float64]  # all in the initial ingoing Cartesian chart
    fate: str
    reliability: str
    charts: tuple[str, ...]
    branches: tuple[int, ...]
    events: tuple["DisplayEvent", ...] = ()
    escape_radius: float | None = None


@dataclass(frozen=True, slots=True)
class DisplayEvent:
    kind: str
    position: NDArray[np.float64]  # canonical ingoing Cartesian, same as ray.positions
    label: str
    affine_parameter: float


@dataclass(frozen=True, slots=True)
class DisplayScene:
    spin: float
    horizon: HorizonInfo
    singularity: SingularityInfo
    rays: tuple[DisplayRay, ...] = ()


def geometry_scene(spin: float, rays=()) -> DisplayScene:
    geometry = KerrSchildGeometry(SpacetimeParameters(a_star=float(spin)))
    return DisplayScene(float(spin), geometry.horizons(), geometry.singularity(), tuple(rays))


def horizon_visible(horizon: HorizonInfo) -> bool:
    return horizon.regime != HorizonRegime.SUPER_EXTREMAL and horizon.exists


def horizon_mesh(horizon: HorizonInfo, spin: float, latitudes=24, longitudes=48):
    """Indexed oblate r=r+ marker; its triangles are a visual surface only."""
    if not horizon_visible(horizon):
        return _readonly(np.empty((0, 3)), dtype=np.float32)
    if latitudes < 3 or longitudes < 3:
        raise ValueError("horizon tessellation must have at least three divisions")
    r = float(horizon.r_plus)
    equatorial = sqrt(r*r + spin*spin)
    vertices = []
    for j in range(latitudes):
        theta0, theta1 = pi*j/latitudes, pi*(j+1)/latitudes
        for k in range(longitudes):
            phi0, phi1 = 2*pi*k/longitudes, 2*pi*(k+1)/longitudes
            def point(theta, phi):
                return (equatorial*sin(theta)*cos(phi),
                        equatorial*sin(theta)*sin(phi), r*cos(theta))
            p00, p10 = point(theta0, phi0), point(theta1, phi0)
            p01, p11 = point(theta0, phi1), point(theta1, phi1)
            vertices.extend((p00, p10, p01, p01, p10, p11))
    return _readonly(vertices, dtype=np.float32)


def singularity_marker(info: SingularityInfo, segments=128):
    """Point at a=0; closed mathematical ring polyline otherwise."""
    if info.kind == "POINT":
        return _readonly([[0., 0., info.plane_z]])
    if info.kind != "RING" or segments < 8:
        raise ValueError("invalid singularity marker definition")
    angles = np.linspace(0., 2*pi, segments+1)
    return _readonly(np.column_stack((info.ring_radius*np.cos(angles),
                                      info.ring_radius*np.sin(angles),
                                      np.full(segments+1, info.plane_z))))


def trajectory_to_display(name: str, result: TrajectoryResult) -> DisplayRay:
    """Convert chart coordinates per sample without interpolating across handoffs.

    A diagnostic failure retains its *sampled prefix* and remains explicitly
    labeled unreliable; the label is never promoted to a physical fate.
    """
    assessment = assess_trajectory(result)
    positions = np.empty((len(result.states), 3), dtype=np.float64)
    geometries = {}
    for i, (state, chart, branch) in enumerate(zip(
            result.states, result.sample_charts, result.sample_branches)):
        key = (chart, int(branch))
        if key not in geometries:
            geometries[key] = KerrSchildGeometry(
                SpacetimeParameters(a_star=result.spin), chart=chart, branch=int(branch))
        positions[i] = geometries[key].canonical_ingoing_position(state[1:4])
    if not np.all(np.isfinite(positions)):
        raise ValueError("non-finite trajectory display coordinates")
    def at_lambda(affine: float):
        # Disk/handoff roots are recorded as exact samples when possible. If
        # the event falls between saved samples, interpolate *only its marker*
        # inside one chart. Never interpolate over a chart discontinuity.
        times = result.affine_parameters
        index = int(np.searchsorted(times, affine, side="left"))
        for j in (index, index-1):
            if 0 <= j < len(times) and abs(times[j]-affine) <= 1e-10:
                return positions[j]
        if 0 < index < len(times) and result.sample_charts[index-1] == result.sample_charts[index]:
            weight = (affine-times[index-1])/(times[index]-times[index-1])
            return (1-weight)*positions[index-1]+weight*positions[index]
        return None  # no invented cross-chart event coordinates

    events = []
    for crossing in result.horizon_crossings:
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=result.spin),
                                      chart=crossing.chart, branch=crossing.branch)
        position = geometry.canonical_ingoing_position(crossing.state[1:4])
        prefix = "OUTER" if crossing.kind.value == "OUTER_HORIZON" else "INNER"
        direction = "I" if crossing.direction == "INWARD" else "O"
        events.append(DisplayEvent(crossing.kind.value, _readonly(position),
                                   f"{prefix}-{direction}", crossing.affine_parameter))
    for transition in result.region_transitions:
        if transition.kind not in ("DISK_CROSSING", "CHART_HANDOFF"):
            continue
        position = at_lambda(transition.affine_parameter)
        if position is not None:
            label = "DISK" if transition.kind == "DISK_CROSSING" else "CHART"
            events.append(DisplayEvent(transition.kind, _readonly(position), label,
                                       transition.affine_parameter))
    events.sort(key=lambda item: item.affine_parameter)
    return DisplayRay(name, _readonly(positions), result.status.value,
                      assessment.level.value, tuple(result.sample_charts),
                      tuple(int(value) for value in result.sample_branches),
                      tuple(events), result.settings.escape_radius)


def ribbon_vertices(positions: NDArray[np.float64]):
    """Two triangles per 3D segment, expanded to fixed pixels in the shader.

    Attributes: start.xyz, end.xyz, endpoint parameter, signed perpendicular.
    Each segment is independent; this retains every recorded sample, including
    regular disk crossings and both signed-r sheets.
    """
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("positions must have shape (N,3) with finite samples")
    if len(points) < 2:
        return _readonly(np.empty((0, 8)), dtype=np.float32)
    records = []
    for start, end in zip(points[:-1], points[1:]):
        if np.array_equal(start, end):
            continue
        for t, side in ((0., -1.), (0., 1.), (1., -1.),
                        (1., -1.), (0., 1.), (1., 1.)):
            records.append((*start, *end, t, side))
    return _readonly(np.reshape(records, (-1, 8)), dtype=np.float32)
