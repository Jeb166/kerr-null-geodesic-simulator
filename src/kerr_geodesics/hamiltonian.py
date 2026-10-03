"""Headless Hamiltonian evaluation and pointwise null-geodesic RHS.

No integrator, initial-condition tetrad, or Kerr-specific equation lives here.
The caller supplies a Geometry and one state ordered
    (t, x, y, z, p_t, p_x, p_y, p_z).
Signature is (-,+,+,+); momenta are covariant. Lambda is affine when the
RHS is later integrated. See the Hamilton equations in README.md.
"""

import numpy as np
from numpy.typing import NDArray

from .geometry import Geometry


def _state8(state: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (8,) or not np.all(np.isfinite(values)):
        raise ValueError("state must have eight finite (t,x,y,z,p_t,p_x,p_y,p_z) entries")
    return values


def hamiltonian_value(geometry: Geometry, state: NDArray[np.float64]) -> float:
    """Return H = 1/2 g^(mu nu) p_mu p_nu; zero means a null state."""
    values = _state8(state)
    inverse = geometry.inverse_metric(values[1:4])
    momentum = values[4:8]
    return float(0.5 * momentum @ inverse @ momentum)


def null_geodesic_rhs(
    geometry: Geometry, state: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate Hamilton's equations at one state; does not integrate it.

    xdot^mu = g^(mu nu) p_nu and
    pdot_mu = -1/2 (partial_mu g^(alpha beta)) p_alpha p_beta.
    The time derivative of a stationary geometry is identically zero.
    This evaluation also works for a non-null state, which is useful in
    derivative tests; the caller must select null initial momentum.
    """
    values = _state8(state)
    position = values[1:4]
    momentum = values[4:8]
    inverse = geometry.inverse_metric(position)
    derivatives = geometry.inverse_metric_derivatives(position)
    xdot = inverse @ momentum
    pdot = -0.5 * np.einsum("mab,a,b->m", derivatives, momentum, momentum)
    return np.concatenate((xdot, pdot))
