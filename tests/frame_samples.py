"""Fixed, reproducible v0.1C frame sample spanning both horizon sides."""

from math import hypot

import numpy as np


SPINS = (0.0, 0.7, -0.8, 0.999, 1.05)
_DIRECTION = np.array([0.28, -0.42, 0.86])
DIRECTIONS = (
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    _DIRECTION / np.linalg.norm(_DIRECTION),
)


def positions_for(geometry):
    a = geometry.parameters.a_star
    outer = geometry.horizons().r_plus
    near_outer = (outer if outer is not None else 1.1) + 0.03
    return (
        np.array([3.0, 1.0, 0.4]),
        np.array([0.0, 0.0, 2.7]),
        np.array([0.8, -0.4, 1.2]),
        np.array([hypot(near_outer, a), 0.0, 0.0]),
        np.array([hypot(0.8, a), 0.0, 0.0]),
    )
