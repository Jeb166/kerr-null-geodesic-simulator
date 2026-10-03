"""Deterministic, renderer-owned ray palette, glyph policy and arrow placement."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RayColor:
    name: str
    hex: str
    rgb: tuple[float, float, float]


RAY_PALETTE = (
    RayColor("Yellow", "#ffe049", (1.0, .879, .286)),
    RayColor("Cyan", "#39e4ff", (.224, .894, 1.0)),
    RayColor("Magenta", "#fa60e5", (.980, .376, .898)),
)


def ray_color(index: int) -> RayColor:
    return RAY_PALETTE[index % len(RAY_PALETTE)]


# Glyph types: 1 START (center dot), 2 open ESCAPED ring, 3 approach dot,
# 4 PARTIAL square, 5 diagnostic X, 6 numerical diamond X, 7 event diamond,
# 8 disk square, 9 chart ring. CANCELLED/SUPERSEDED have no physical endpoint.
ENDPOINT_GLYPHS = {
    "ESCAPED": 2, "SINGULARITY_APPROACH": 3,
    "MAX_INTEGRATION_LIMIT": 4, "CHART_BOUNDARY": 4,
    "DIAGNOSTIC_FAILURE": 5, "NUMERICAL_ERROR": 6,
}
EVENT_GLYPHS = {
    "OUTER_HORIZON": (7, (0.30, .59, 1.0)),
    "INNER_HORIZON": (7, (.69, .43, 1.0)),
    "DISK_CROSSING": (8, (.94, .96, .99)),
    "CHART_HANDOFF": (9, (1.0, .56, .17)),
}


def ray_opacity(reliability: str) -> float:
    return {"COMPLETE": 1.0, "PARTIAL": .64, "UNRELIABLE": .40}.get(reliability, .40)


def endpoint_glyph(fate: str):
    return ENDPOINT_GLYPHS.get(fate)  # compute cancellation has no physics endpoint


def termination_reason(fate: str, escape_radius: float | None = None) -> str:
    if fate == "ESCAPED":
        return ("reached finite escape boundary"
                + (f" (R_escape={escape_radius:g})" if escape_radius is not None else ""))
    return {
        "SINGULARITY_APPROACH": "Sigma threshold reached; ring/point is not claimed",
        "MAX_INTEGRATION_LIMIT": "integration budget reached; fate undetermined",
        "CHART_BOUNDARY": "chart domain ended; fate undetermined",
        "DIAGNOSTIC_FAILURE": "null/conserved reliability gate failed",
        "NUMERICAL_ERROR": "integrator failed; fate undetermined",
        "CANCELLED": "compute cancelled; no physical endpoint",
        "SUPERSEDED": "stale compute request; no physical endpoint",
    }.get(fate, "no physical endpoint classification")


@dataclass(frozen=True, slots=True)
class MarkerSpec:
    position: tuple[float, float, float]
    glyph: int
    rgba: tuple[float, float, float, float]
    radius: float


def ray_markers(ray, index: int):
    """Marker data from existing display samples/events, never propagated."""
    if not len(ray.positions):
        return ()
    rgb = ray_color(index).rgb
    markers = [MarkerSpec(tuple(ray.positions[0]), 1, (*rgb, 1.), 9.)]
    end = endpoint_glyph(ray.fate)
    if end is not None:
        markers.append(MarkerSpec(tuple(ray.positions[-1]), end, (*rgb, 1.), 10.))
    for event in ray.events:
        if event.kind in EVENT_GLYPHS:
            glyph, color = EVENT_GLYPHS[event.kind]
            markers.append(MarkerSpec(tuple(event.position), glyph, (*color, 1.), 5.))
    return tuple(markers)


@dataclass(frozen=True, slots=True)
class DirectionArrow:
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    fraction: float  # tip along recorded segment; no new physics sample


def direction_arrows(positions, count=4):
    """Sparse tangent arrows along recorded polyline in original sample order."""
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("expected finite (N,3) trajectory")
    if len(points) < 2 or count <= 0:
        return ()
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    if cumulative[-1] == 0:
        return ()
    chosen = []
    for progress in np.linspace(.13, .87, count):
        target = progress*cumulative[-1]
        segment = min(int(np.searchsorted(cumulative, target, side="right"))-1, len(lengths)-1)
        if lengths[segment] <= 0:
            continue
        fraction = (target-cumulative[segment])/lengths[segment]
        chosen.append(DirectionArrow(tuple(points[segment]), tuple(points[segment+1]),
                                     float(fraction)))
    return tuple(chosen)
