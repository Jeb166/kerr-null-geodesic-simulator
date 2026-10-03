"""Independent, test-only numerical checks; never imported by src/.

Central differences differentiate inverse_metric and hamiltonian_value by
re-evaluating each at displaced coordinates while keeping p_mu fixed.
They do not call the analytic derivative method or Hamiltonian RHS.
"""

import numpy as np

from kerr_geodesics import hamiltonian_value


FD_FACTOR = 1e-5
POSITIONS = (
    np.array([3.1, -1.7, 0.8]),
    np.array([0.9, 0.2, 0.7]),
    np.array([1.8, 0.0, 0.0]),
    np.array([0.0, 0.0, 2.5]),
    np.array([-2.1, 1.3, -0.5]),
    np.array([4.0, -3.0, 1.1]),
    np.array([-1.2, -0.9, 2.2]),
)
SPINS = (0.0, 0.7, -0.8, 0.999)
SPATIAL_MOMENTA = (
    np.array([0.3, -0.6, 0.8]),
    np.array([-0.5, 0.2, 0.4]),
)


def step(coordinate):
    return FD_FACTOR * max(1.0, abs(float(coordinate)))


def finite_difference_inverse_metric(geometry, position):
    result = np.zeros((4, 4, 4), dtype=np.float64)
    for axis in range(3):
        h = step(position[axis])
        delta = np.zeros(3)
        delta[axis] = h
        result[axis + 1] = (
            geometry.inverse_metric(position + delta)
            - geometry.inverse_metric(position - delta)
        ) / (2.0 * h)
    # No time argument in the stationary geometry API.
    return result


def finite_difference_hamiltonian_gradient(geometry, state):
    gradient = np.zeros(4, dtype=np.float64)
    for axis in range(4):
        h = step(state[axis])
        plus = state.copy()
        minus = state.copy()
        plus[axis] += h
        minus[axis] -= h
        gradient[axis] = (
            hamiltonian_value(geometry, plus)
            - hamiltonian_value(geometry, minus)
        ) / (2.0 * h)
    return gradient


def controlled_null_state(geometry, position, spatial_momentum):
    """Test-only: solve g^munu p_mu p_nu=0 for p_t, choosing k^t>0.

    With A=g^tt, B=g^ti p_i, C_quad=g^ij p_i p_j,
    A*p_t^2 + 2*B*p_t + C_quad=0. The +sqrt root for the numerator
    makes k^t=A*p_t+B positive. No camera or tetrad is involved.
    """
    inverse = geometry.inverse_metric(position)
    A = inverse[0, 0]
    B = inverse[0, 1:] @ spatial_momentum
    C_quad = spatial_momentum @ inverse[1:, 1:] @ spatial_momentum
    root = float(np.sqrt(B * B - A * C_quad))
    p_t = (-B + root) / A
    state = np.r_[0.25, position, p_t, spatial_momentum]
    assert (inverse @ state[4:])[0] > 0.0
    return state
