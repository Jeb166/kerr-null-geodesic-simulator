"""Reproducible maximum errors for the fixed v0.1B test sample.

Test-only finite-difference reference lives in tests/reference_calculations.py;
production src/ code never imports it. Run after installing the project.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from reference_calculations import (  # noqa: E402 - test-only path above
    FD_FACTOR,
    POSITIONS,
    SPATIAL_MOMENTA,
    SPINS,
    controlled_null_state,
    finite_difference_hamiltonian_gradient,
    finite_difference_inverse_metric,
)

from kerr_geodesics import (  # noqa: E402
    KerrSchildGeometry,
    SpacetimeParameters,
    hamiltonian_value,
    null_geodesic_rhs,
)


def main():
    max_derivative = 0.0
    max_momentum_rhs = 0.0
    max_stationarity = 0.0
    max_null = 0.0
    for spin in SPINS:
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=spin))
        for position in POSITIONS:
            max_derivative = max(
                max_derivative,
                float(np.max(np.abs(
                    geometry.inverse_metric_derivatives(position)
                    - finite_difference_inverse_metric(geometry, position)
                ))),
            )
            for momentum in SPATIAL_MOMENTA:
                state = controlled_null_state(geometry, position, momentum)
                rhs = null_geodesic_rhs(geometry, state)
                max_null = max(max_null, abs(hamiltonian_value(geometry, state)))
                max_stationarity = max(max_stationarity, abs(float(rhs[4])))
                max_momentum_rhs = max(
                    max_momentum_rhs,
                    float(np.max(np.abs(
                        rhs[4:] + finite_difference_hamiltonian_gradient(geometry, state)
                    ))),
                )
    print(f"Python: {sys.version.split()[0]}; NumPy: {np.__version__}")
    print(f"spins={SPINS}; positions={len(POSITIONS)}; momenta_per_position={len(SPATIAL_MOMENTA)}")
    print(f"fd_step_factor={FD_FACTOR:g} * max(1, abs(coordinate))")
    print(f"max_abs_inverse_metric_derivative_error={max_derivative:.12e}")
    print(f"max_abs_momentum_rhs_error={max_momentum_rhs:.12e}")
    print(f"max_abs_dp_t_dlambda={max_stationarity:.12e}")
    print(f"max_abs_null_hamiltonian={max_null:.12e}")
    if max_derivative >= 2e-8 or max_momentum_rhs >= 2e-8 or max_null >= 2e-15:
        raise SystemExit("v0.1B diagnostics FAILED")


if __name__ == "__main__":
    main()
