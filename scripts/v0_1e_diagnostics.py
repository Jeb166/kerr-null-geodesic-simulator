"""Reproduce v0.1E signed-sheet FD checks and selected real trajectories.

Run after editable install: python scripts/v0_1e_diagnostics.py
All FD code is imported from tests, never from the production package.
"""

from pathlib import Path
import sys

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from reference_calculations import (  # noqa: E402
    FD_FACTOR, POSITIONS, SPATIAL_MOMENTA, SPINS, controlled_null_state,
    finite_difference_hamiltonian_gradient, finite_difference_inverse_metric,
)

from kerr_geodesics import (  # noqa: E402
    IntegrationSettings, KerrSchildGeometry, RayDefinition,
    SpacetimeParameters, integrate_ray, null_geodesic_rhs,
)


def numerical_validation():
    derivative_error = rhs_error = inverse_error = determinant_error = 0.0
    max_stationarity = 0.0
    cases = 0
    for spin in SPINS:
        g = KerrSchildGeometry(SpacetimeParameters(a_star=spin))
        points = POSITIONS
        for point in points:
            cases += 1
            derivative_error = max(derivative_error, np.max(np.abs(
                g.inverse_metric_derivatives(point)-finite_difference_inverse_metric(g, point))))
            for momentum in SPATIAL_MOMENTA:
                state = controlled_null_state(g, point, momentum)
                rhs = null_geodesic_rhs(g, state)
                rhs_error = max(rhs_error, np.max(np.abs(
                    rhs[4:]+finite_difference_hamiltonian_gradient(g, state))))
                max_stationarity = max(max_stationarity, abs(rhs[4]))

    for spin in (.7, .999, 1.05, 1.2):
        for chart in ("ingoing", "outgoing"):
            g = KerrSchildGeometry(SpacetimeParameters(a_star=spin),
                                   branch=-1, chart=chart)
            for point in map(np.array, ([.1,.05,.18], [.4,.15,-.6], [3.,-1.,.4])):
                cases += 1
                derivative_error = max(derivative_error, np.max(np.abs(
                    g.inverse_metric_derivatives(point)-finite_difference_inverse_metric(g,point))))
                inverse_error = max(inverse_error, np.max(np.abs(g.metric(point)@g.inverse_metric(point)-np.eye(4))))
                determinant_error = max(determinant_error, abs(np.linalg.det(g.metric(point))+1.))
                # Null momenta are chosen on negative sheets where g^tt<0.
                if (spin in (.7,1.05) and
                        (np.allclose(point,[.1,.05,.18]) or np.allclose(point,[3.,-1.,.4]))):
                    state=controlled_null_state(g,point,np.array([.3,-.2,.4]))
                    rhs=null_geodesic_rhs(g,state)
                    rhs_error=max(rhs_error,np.max(np.abs(rhs[4:]+finite_difference_hamiltonian_gradient(g,state))))
                    max_stationarity=max(max_stationarity,abs(rhs[4]))
    for hemisphere in (-1,1):
        g=KerrSchildGeometry(SpacetimeParameters(a_star=.8),disk_hemisphere=hemisphere)
        point=np.array([.2,.1,0.]); cases+=1
        derivative_error=max(derivative_error,np.max(np.abs(
            g.inverse_metric_derivatives(point)-finite_difference_inverse_metric(g,point))))

    print(f"Python={sys.version.split()[0]}, NumPy={np.__version__}, SciPy={scipy.__version__}")
    print(f"FD: h_i={FD_FACTOR:g}*max(1,abs(x_i)); points={cases}; tolerances derivative/RHS=2e-8")
    print(f"max derivative absolute error={derivative_error:.12e}")
    print(f"max Hamiltonian-gradient/RHS absolute error={rhs_error:.12e}")
    print(f"max |dp_t/dlambda|={max_stationarity:.12e}")
    print(f"negative-sheet max |g*g^-1-I|={inverse_error:.12e}; max |det(g)+1|={determinant_error:.12e}")


def scenario(name, spin, position, direction, branch=1):
    g=KerrSchildGeometry(SpacetimeParameters(a_star=spin),branch=branch)
    settings=IntegrationSettings(escape_radius=8.,max_affine_parameter=35.)
    result=integrate_ray(g,RayDefinition(position,direction),settings)
    print(f"[{name}] spin={spin}, x0={position}, d_local={direction}, initial_branch={branch}")
    print(f" status={result.status.value}, lambda={result.affine_parameter_end:.12f}, "
          f"steps={result.accepted_steps}, nfev={result.rhs_evaluations}, "
          f"min_step={result.minimum_accepted_step:.6e}, rejected_steps={result.rejected_steps}")
    print(f" min_abs_r={result.minimum_sampled_abs_r:.12e}, min_sigma={result.minimum_sampled_sigma:.12e}, "
          f"scaled_null={result.max_scaled_null_residual:.12e}, raw_null={result.max_abs_null_residual:.12e}")
    print(f" max_Lz_drift={result.max_abs_Lz_drift:.12e}, "
          f"max_Carter_drift={result.max_abs_carter_constant_drift:.12e}, "
          f"scaled_Carter_drift={result.max_scaled_carter_constant_drift:.12e}")
    print(" transitions="+str([(t.kind,round(t.affine_parameter,9),t.from_region,
                                t.to_region,t.from_chart,t.to_chart)
                              for t in result.region_transitions]))
    print(" horizons="+str([(e.kind.value,e.direction,round(e.affine_parameter,9),e.chart)
                           for e in result.horizon_crossings]))
    print(f" final_region={result.sample_regions[-1]}, final_chart={result.sample_charts[-1]}")
    if result.status.value in ("NUMERICAL_ERROR","DIAGNOSTIC_FAILURE"):
        print(f" diagnostic={result.message}")


if __name__ == "__main__":
    numerical_validation()
    scenario("previous inclined failure", .7,[4.,.5,.6],[-1.,0.,0.])
    scenario("regular off-axis disk crossing",.8,[.4,.1,3.],[0.,0.,-1.])
    scenario("regular axis disk crossing",.8,[0.,0.,3.],[0.,0.,-1.])
    scenario("negative-to-positive disk crossing",.8,[0.,0.,-.1],[0.,0.,1.],-1)
    scenario("super-extremal close passage",1.2,[1.3,.1,3.],[0.,0.,-1.])
    scenario("super-extremal second case",1.05,[1.,.1,3.],[0.,0.,-1.])
