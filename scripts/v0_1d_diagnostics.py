"""Selected real-ray results and two-level tolerance convergence."""

import sys
from math import sqrt

import numpy as np
import scipy

from kerr_geodesics import (
    IntegrationSettings,
    KerrSchildGeometry,
    RayDefinition,
    SpacetimeParameters,
    TerminationStatus,
    eulerian_frame,
    integrate_ray,
)


def scenarios():
    yield "Schwarzschild outward", 0.0, [4, 0, 0], [1, 0, 0]
    yield "Schwarzschild inward", 0.0, [4, 0, 0], [-1, 0, 0]
    yield "Kerr equatorial plunge", 0.7, [4, 0, 0], [-1, 0, 0]
    yield "Kerr regular axis disk", 0.8, [0, 0, 3], [0, 0, -1]
    yield "Near-extremal exterior", 0.999, [4, 0.1, 0.5], [1, 0, 0]
    yield "Super-extremal exterior", 1.05, [4, 0.1, 0.5], [1, 0, 0]
    vec = np.array([.7, .2, .35])
    yield "Kerr inclined exterior", 0.7, [4, 1, .6], vec / np.linalg.norm(vec)
    # Historical v0.1D failure; in v0.1E an outgoing patch is selected after
    # the inner radial turn. See v0_1e_diagnostics.py for branch metadata.
    yield "Kerr formerly failing inclined inner branch", 0.7, [4, .5, .6], [-1, 0, 0]


def main():
    print(f"Python={sys.version.split()[0]} NumPy={np.__version__} SciPy={scipy.__version__}")
    for name, spin, xyz, direction in scenarios():
        g = KerrSchildGeometry(SpacetimeParameters(a_star=spin))
        ray = RayDefinition(xyz, direction)
        settings = IntegrationSettings(escape_radius=9.0 if name == "Kerr inclined exterior" else 8.0,
                                       max_affine_parameter=35.0)
        result = integrate_ray(g, ray, settings)
        c = result.initial.conserved
        events = ",".join(
            f"{event.kind.value}@{event.affine_parameter:.9f}({event.direction})"
            for event in result.horizon_crossings
        ) or "none"
        print(f"[{name}] a={spin:+.3f} x0={xyz} d_local={np.asarray(direction).round(6).tolist()}")
        print(f" E={c.energy:+.10f} L_z={c.angular_momentum_z:+.10f} "
              f"carter_constant={c.carter_constant:+.10f}")
        print(f" status={result.status.value} lambda={result.affine_parameter_end:.12f} "
              f"steps={result.accepted_steps} rejected={result.rejected_steps} nfev={result.rhs_evaluations}")
        print(f" horizons={events} min_sampled_r={result.minimum_sampled_r:.9g} "
              f"min_sampled_sigma={result.minimum_sampled_sigma:.9g}")
        print(f" max_null_raw={result.max_abs_null_residual:.6e} "
              f"max_null_scaled={result.max_scaled_null_residual:.6e} "
              f"L_z_drift={result.max_abs_Lz_drift:.6e} "
              f"carter_constant_drift={result.max_abs_carter_constant_drift:.6e}")
        if result.status == TerminationStatus.NUMERICAL_ERROR:
            print(f" diagnostic={result.message}")

    for spin, xyz, direction in (
        (0.0, [4, 0, 0], [1, 0, 0]),
        (0.7, [4, 0, 0], [-1, 0, 0]),
    ):
        g = KerrSchildGeometry(SpacetimeParameters(a_star=spin))
        ray = RayDefinition(xyz, direction)
        results = [integrate_ray(g, ray, IntegrationSettings(escape_radius=8, rtol=tol))
                   for tol in (1e-9, 1e-11)]
        loose, tight = results
        print(f"[convergence a={spin}] rtol=1e-9/1e-11 "
              f"status={loose.status.value}/{tight.status.value} "
              f"delta_lambda={abs(loose.affine_parameter_end-tight.affine_parameter_end):.6e} "
              f"max_delta_final_state={np.max(np.abs(loose.states[-1]-tight.states[-1])):.6e} "
              f"null_scaled={loose.max_scaled_null_residual:.6e}/{tight.max_scaled_null_residual:.6e}")

    g = KerrSchildGeometry(SpacetimeParameters(a_star=0))
    xyz = np.array([3., 0., 0.])
    p = g.metric(xyz) @ np.array([3., 0., sqrt(3), 0.])
    frame = eulerian_frame(g, xyz)
    direction = (frame.triad @ p) / (-p @ frame.observer)
    orbit = integrate_ray(g, RayDefinition(xyz, direction),
                          IntegrationSettings(escape_radius=8, max_affine_parameter=2, max_step=.1))
    print(f"[photon sphere] L_z/E={orbit.initial.conserved.angular_momentum_z/orbit.initial.conserved.energy:.12f} "
          f"max_abs_r_minus_3={max(abs(g.radial_coordinate(s[1:4])-3) for s in orbit.states):.6e} "
          f"status={orbit.status.value}")


if __name__ == "__main__":
    main()
