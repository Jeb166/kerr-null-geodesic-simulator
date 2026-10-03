"""Small reproducible numerical report for v0.1A geometry only."""

import numpy as np

from kerr_geodesics import KerrSchildGeometry, SpacetimeParameters


def main() -> None:
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(5903)
    points = rng.uniform(-5.0, 5.0, size=(200, 3))
    max_identity = 0.0
    max_ell_null = 0.0
    max_det = 0.0
    count = 0
    for spin in (0.0, 0.8, 0.9999, 1.0, 1.05):
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=spin))
        for point in points:
            _, ell = geometry.kerr_schild_fields(point)
            g = geometry.metric(point)
            g_inv = geometry.inverse_metric(point)
            max_identity = max(max_identity, np.max(np.abs(g @ g_inv - np.eye(4))))
            max_ell_null = max(max_ell_null, abs(ell @ eta @ ell))
            max_det = max(max_det, abs(np.linalg.det(g) + 1))
            count += 1
        h = geometry.horizons()
        print(f"a_star={spin:.4f}: regime={h.regime.value}; r_plus={h.r_plus}; r_minus={h.r_minus}")
    print(f"samples={count}")
    print(f"max_abs_metric_times_inverse_minus_identity={max_identity:.6e}")
    print(f"max_abs_eta_null_ell={max_ell_null:.6e}")
    print(f"max_abs_det_g_plus_one={max_det:.6e}")
    if max_identity > 2e-12 or max_ell_null > 2e-12 or max_det > 2e-11:
        raise SystemExit("Geometry validation FAILED")


if __name__ == "__main__":
    main()
