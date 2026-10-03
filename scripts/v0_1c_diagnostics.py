"""Fixed local-frame measurements; no trajectory integration."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from frame_samples import DIRECTIONS, SPINS, positions_for  # noqa: E402

from kerr_geodesics import (  # noqa: E402
    KerrSchildGeometry,
    RayDefinition,
    SpacetimeParameters,
    create_initial_conditions,
    hamiltonian_value,
)


def main():
    errors = dict(normal=0.0, triad=0.0, orthogonal=0.0,
                  tangent_null=0.0, momentum_null=0.0,
                  hamiltonian=0.0, local_energy=0.0)
    sample_count = 0
    positions_count = 0
    for spin in SPINS:
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=spin))
        for position in positions_for(geometry):
            positions_count += 1
            metric = geometry.metric(position)
            inverse = geometry.inverse_metric(position)
            for direction in DIRECTIONS:
                sample_count += 1
                result = create_initial_conditions(geometry, RayDefinition(position, direction))
                n, e, k, p = result.frame.observer, result.frame.triad, result.tangent, result.momentum
                errors["normal"] = max(errors["normal"], abs(float(n @ metric @ n + 1)))
                errors["triad"] = max(errors["triad"], float(np.max(np.abs(e @ metric @ e.T - np.eye(3)))))
                errors["orthogonal"] = max(errors["orthogonal"], float(np.max(np.abs(e @ metric @ n))))
                errors["tangent_null"] = max(errors["tangent_null"], abs(float(k @ metric @ k)))
                errors["momentum_null"] = max(errors["momentum_null"], abs(float(p @ inverse @ p)))
                errors["hamiltonian"] = max(errors["hamiltonian"], abs(hamiltonian_value(geometry, result.state)))
                errors["local_energy"] = max(errors["local_energy"], abs(result.local_energy - 1))
        representative = create_initial_conditions(
            geometry, RayDefinition(np.array([3.0, 1.0, 0.4]), DIRECTIONS[-1])
        )
        conserved = representative.conserved
        print(f"spin={spin:+.3f}: E={conserved.energy:+.12f} "
              f"L_z={conserved.angular_momentum_z:+.12f} "
              f"carter_constant={conserved.carter_constant:+.12f}")
    print(f"spins={len(SPINS)} positions={positions_count} directions={len(DIRECTIONS)} samples={sample_count}")
    for name, error in errors.items():
        print(f"max_abs_{name}_error={error:.12e}")
    if max(errors.values()) >= 3e-12:
        raise SystemExit("v0.1C frame diagnostics FAILED")


if __name__ == "__main__":
    main()
