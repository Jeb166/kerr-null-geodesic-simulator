"""v0.1E: signed Kerr sheets, chart handoffs and measured inner orbits."""

import math
from pathlib import Path
import tomllib
import unittest

import numpy as np
from scipy.integrate import quad

from kerr_geodesics import (
    ConservedQuantities, DomainKind, EventKind, IntegrationSettings, KerrSchildGeometry,
    RayDefinition, SpacetimeParameters, TerminationStatus, hamiltonian_value,
    integrate_ray, null_geodesic_rhs,
)
from reference_calculations import (
    controlled_null_state, finite_difference_hamiltonian_gradient,
    finite_difference_inverse_metric,
)


def geometry(spin, **kwargs):
    return KerrSchildGeometry(SpacetimeParameters(a_star=spin), **kwargs)


class SignedKerrGeometryTests(unittest.TestCase):
    def test_negative_sheet_identities_in_both_horizon_patches(self):
        eta = np.diag([-1., 1., 1., 1.])
        positions = ([.1, .05, .18], [.4, .15, -.6], [3., -1., .4])
        for spin in (.7, .999, 1.05, 1.2):
            for chart in ("ingoing", "outgoing"):
                g = geometry(spin, branch=-1, chart=chart)
                for position in positions:
                    with self.subTest(spin=spin, chart=chart, position=position):
                        pos = np.array(position)
                        info = g.domain(pos)
                        self.assertEqual(info.kind, DomainKind.REGULAR)
                        self.assertLess(info.r, 0.)
                        self.assertGreater(info.sigma, 0.)
                        H, ell = g.kerr_schild_fields(pos)
                        self.assertLess(H, 0.)
                        self.assertLess(abs(ell @ eta @ ell), 5e-15)
                        self.assertLess(np.max(abs(g.metric(pos) @ g.inverse_metric(pos)-np.eye(4))), 2e-13)
                        self.assertLess(abs(np.linalg.det(g.metric(pos))+1.), 2e-13)

    def test_negative_and_exact_disk_derivatives_against_independent_fd(self):
        positions = ([.1, .05, .18], [.4, .15, -.6], [3., -1., .4])
        for spin in (.7, .999, 1.05, 1.2):
            for chart in ("ingoing", "outgoing"):
                g = geometry(spin, branch=-1, chart=chart)
                for position in positions:
                    with self.subTest(spin=spin, chart=chart, position=position):
                        p = np.array(position)
                        analytic = g.inverse_metric_derivatives(p)
                        self.assertTrue(np.array_equal(analytic[0], np.zeros((4, 4))))
                        self.assertLess(np.max(abs(analytic-finite_difference_inverse_metric(g,p))), 2e-8)
        for hemisphere in (-1, 1):
            g = geometry(.8, disk_hemisphere=hemisphere)
            pos = np.array([.2, .1, 0.])
            self.assertEqual(g.domain(pos).kind, DomainKind.REGULAR)
            self.assertEqual(g.radial_coordinate(pos), 0.)
            self.assertAlmostEqual(g.domain(pos).sigma, .8**2-.2**2-.1**2)
            self.assertLess(np.max(abs(g.inverse_metric_derivatives(pos)
                                       -finite_difference_inverse_metric(g,pos))), 2e-8)

    def test_negative_sheet_null_rhs_and_gradient(self):
        for spin in (.7, 1.05):
            for chart in ("ingoing", "outgoing"):
                g = geometry(spin, branch=-1, chart=chart)
                for position in ([.1, .05, .18], [3., -1., .4]):
                    state = controlled_null_state(g, np.array(position), np.array([.3,-.2,.4]))
                    rhs = null_geodesic_rhs(g, state)
                    self.assertLess(abs(hamiltonian_value(g, state)), 2e-15)
                    self.assertEqual(rhs[4], 0.)
                    np.testing.assert_allclose(rhs[:4], g.inverse_metric(state[1:4]) @ state[4:], atol=2e-15, rtol=0)
                    self.assertLess(np.max(abs(rhs[4:]+finite_difference_hamiltonian_gradient(g,state))), 2e-8)

    def test_canonical_handoff_matches_independent_chart_jacobian(self):
        a = .7
        g = geometry(a)
        position = np.array([.12, .08, .22])
        radius = g.radial_coordinate(position)
        self.assertLess(radius, g.horizons().r_minus)
        state = controlled_null_state(g, position, np.array([.3, -.4, .5]))
        outgoing, switched = g.horizon_chart_handoff(state, "outgoing")
        self.assertEqual(outgoing.chart, "outgoing")

        def map_coordinates(coords):
            r = g.radial_coordinate(coords[1:4])
            delta = lambda s: s*s-2*s+a*a
            dt = -4*quad(lambda s: s/delta(s), radius, r, epsabs=1e-13)[0]
            angle = -2*math.atan2(a,r)-2*a*quad(lambda s: 1/delta(s), radius, r, epsabs=1e-13)[0]
            co, si = math.cos(angle), math.sin(angle)
            x,y,z = coords[1:4]
            return np.array([coords[0]+dt, co*x-si*y, si*x+co*y, z])

        coordinate = state[:4]
        h = 1e-6
        jac = np.column_stack([(map_coordinates(coordinate+h*np.eye(4)[axis])
                                -map_coordinates(coordinate-h*np.eye(4)[axis]))/(2*h)
                               for axis in range(4)])
        np.testing.assert_allclose(switched[:4], map_coordinates(coordinate), atol=2e-14, rtol=0)
        new_inverse = outgoing.inverse_metric(switched[1:4])
        np.testing.assert_allclose(jac @ g.inverse_metric(position) @ jac.T, new_inverse, atol=2e-9, rtol=0)
        np.testing.assert_allclose(new_inverse @ switched[4:8], jac @ g.inverse_metric(position) @ state[4:8], atol=2e-9, rtol=0)
        self.assertLess(abs(hamiltonian_value(outgoing, switched)), 1e-12)
        one = g.conserved_quantities(position,state[4:])
        two = outgoing.conserved_quantities(switched[1:4],switched[4:])
        np.testing.assert_allclose((one.energy,one.angular_momentum_z,one.carter_constant),
                                   (two.energy,two.angular_momentum_z,two.carter_constant),atol=2e-12,rtol=0)

    def test_ring_is_singular_but_disk_is_regular_with_hemisphere(self):
        for branch in (-1,1):
            g=geometry(.8,branch=branch,disk_hemisphere=1)
            self.assertEqual(g.domain(np.array([.8,0.,0.])).kind,DomainKind.SINGULARITY)
            self.assertEqual(g.domain(np.array([0.,0.,0.])).kind,DomainKind.REGULAR)
            np.testing.assert_array_equal(g.metric(np.zeros(3)), np.diag([-1.,1.,1.,1.]))
        # Without a hemisphere, the exact double-covered disk has no unique
        # derivative. This is a chart-data boundary, never a singularity.
        self.assertEqual(geometry(.8).domain(np.array([.4,0.,0.])).kind, DomainKind.CHART_BOUNDARY)


class RegionTrajectoryTests(unittest.TestCase):
    def test_problematic_inner_inclined_ray_is_resolved_in_outgoing_chart(self):
        g=geometry(.7)
        settings=IntegrationSettings(escape_radius=8.,max_affine_parameter=35.)
        result=integrate_ray(g,RayDefinition([4.,.5,.6],[-1.,0.,0.]),settings)
        self.assertEqual(result.status,TerminationStatus.ESCAPED, result.message)
        self.assertEqual([event.kind for event in result.horizon_crossings],
                         [EventKind.OUTER_HORIZON,EventKind.INNER_HORIZON,
                          EventKind.INNER_HORIZON,EventKind.OUTER_HORIZON])
        self.assertEqual([event.direction for event in result.horizon_crossings],
                         ["INWARD","INWARD","OUTWARD","OUTWARD"])
        self.assertEqual([event.chart for event in result.horizon_crossings],
                         ["ingoing","ingoing","outgoing","outgoing"])
        self.assertEqual([transition.kind for transition in result.region_transitions].count("CHART_HANDOFF"), 1)
        self.assertGreater(result.minimum_sampled_sigma,.13)
        self.assertGreater(result.minimum_sampled_abs_r,.21)
        self.assertLess(result.max_scaled_null_residual,2e-9)
        self.assertLess(result.max_abs_Lz_drift,1e-8)
        self.assertLess(result.max_abs_carter_constant_drift,1e-8)
        self.assertEqual(len(result.sample_branches),len(result.states))
        self.assertEqual(len(result.sample_charts),len(result.states))
        self.assertEqual(len(result.sample_regions),len(result.states))
        self.assertIn("INNER",result.sample_regions)
        self.assertEqual(result.sample_regions[-1],"EXTERIOR")

    def test_inclined_ray_tighter_tolerance_converges(self):
        g=geometry(.7); ray=RayDefinition([4.,.5,.6],[-1.,0.,0.])
        results=[integrate_ray(g,ray,IntegrationSettings(escape_radius=8.,rtol=tol))
                 for tol in (1e-9,1e-11)]
        loose,tight=results
        self.assertEqual((loose.status,tight.status),(TerminationStatus.ESCAPED,)*2)
        self.assertEqual(len(loose.horizon_crossings),len(tight.horizon_crossings))
        for one,two in zip(loose.horizon_crossings,tight.horizon_crossings):
            self.assertLess(abs(one.affine_parameter-two.affine_parameter),2e-7)
        self.assertLess(abs(loose.affine_parameter_end-tight.affine_parameter_end),2e-7)
        self.assertLess(np.max(abs(loose.states[-1]-tight.states[-1])),2e-5)

    def test_off_axis_regular_disk_crossing_and_negative_branch(self):
        g=geometry(.8)
        result=integrate_ray(g,RayDefinition([.4,.1,3.],[0.,0.,-1.]),IntegrationSettings(escape_radius=8.))
        self.assertEqual(result.status,TerminationStatus.ESCAPED,result.message)
        self.assertEqual([t.kind for t in result.region_transitions].count("DISK_CROSSING"),1)
        self.assertLess(result.minimum_sampled_abs_r,1e-12)
        self.assertGreater(result.minimum_sampled_sigma,.5)
        self.assertEqual(result.sample_branches[-1],-1)
        self.assertLess(result.max_scaled_null_residual,1e-9)
        self.assertLess(result.max_scaled_carter_constant_drift,1e-9)

    def test_negative_sheet_launch_can_cross_disk_and_outward_horizons(self):
        g=geometry(.8,branch=-1)
        result=integrate_ray(g,RayDefinition([0.,0.,-.1],[0.,0.,1.]),
                             IntegrationSettings(escape_radius=8.))
        self.assertEqual(result.status,TerminationStatus.ESCAPED,result.message)
        self.assertEqual([t.kind for t in result.region_transitions],
                         ["DISK_CROSSING","CHART_HANDOFF","INNER_HORIZON","OUTER_HORIZON"])
        self.assertEqual([e.direction for e in result.horizon_crossings],["OUTWARD","OUTWARD"])
        self.assertEqual((result.sample_branches[0],result.sample_branches[-1]),(-1,1))
        self.assertLess(result.max_scaled_null_residual,2e-9)

    def test_true_equatorial_ring_approach_is_not_disk_crossing(self):
        g=geometry(.7)
        result=integrate_ray(g,RayDefinition([4.,0.,0.],[-1.,0.,0.]),
                             IntegrationSettings(escape_radius=8.,epsilon_sigma=1e-4))
        self.assertEqual(result.status,TerminationStatus.SINGULARITY_APPROACH)
        self.assertAlmostEqual(result.minimum_sampled_sigma,1e-4,delta=2e-14)
        self.assertNotIn("DISK_CROSSING",[t.kind for t in result.region_transitions])

    def test_super_extremal_close_passage_and_double_disk_crossing(self):
        cases=((1.05,1.),(1.2,1.3))
        for spin,x in cases:
            with self.subTest(spin=spin):
                g=geometry(spin)
                result=integrate_ray(g,RayDefinition([x,.1,3.],[0.,0.,-1.]),
                                     IntegrationSettings(escape_radius=8.))
                self.assertEqual(result.status,TerminationStatus.ESCAPED,result.message)
                self.assertFalse(g.horizons().exists)
                self.assertEqual(result.horizon_crossings,())
                self.assertGreater(result.minimum_sampled_sigma,.2)
                self.assertLess(result.minimum_sampled_sigma,.42)
                self.assertLess(result.minimum_sampled_abs_r,1e-11)
                self.assertGreaterEqual([t.kind for t in result.region_transitions].count("DISK_CROSSING"),1)
                self.assertLess(result.max_scaled_null_residual,2e-9)
                self.assertLess(result.max_scaled_Lz_drift,2e-9)
                self.assertLess(result.max_scaled_carter_constant_drift,2e-9)

    def test_extremal_horizon_regime_survives_signed_disk_extension(self):
        g=geometry(1.)
        result=integrate_ray(g,RayDefinition([0.,0.,3.],[0.,0.,-1.]),
                             IntegrationSettings(escape_radius=8.))
        self.assertEqual(result.status,TerminationStatus.ESCAPED,result.message)
        self.assertEqual([e.kind for e in result.horizon_crossings],[EventKind.OUTER_HORIZON])
        self.assertIn("DISK_CROSSING",[t.kind for t in result.region_transitions])

    def test_diagnostic_gate_reports_invariant_corruption(self):
        class DriftingConservedGeometry(KerrSchildGeometry):
            def conserved_quantities(self, position, momentum):
                exact = super().conserved_quantities(position, momentum)
                # Deterministic injected invariant corruption checks the
                # reporting contract without a SciPy-version-dependent ray.
                if position[0] > 4.2:
                    return ConservedQuantities(exact.energy, exact.angular_momentum_z,
                                               exact.carter_constant + 1e-4)
                return exact

        g=DriftingConservedGeometry(SpacetimeParameters(a_star=0.7))
        result=integrate_ray(g,RayDefinition([4.,0.,0.],[1.,0.,0.]),
                             IntegrationSettings(escape_radius=8.))
        self.assertEqual(result.status,TerminationStatus.DIAGNOSTIC_FAILURE)
        self.assertIn("reliability gate exceeded",result.message)
        self.assertGreater(result.max_scaled_carter_constant_drift,1e-8)
        self.assertEqual(result.horizon_crossings,())

    def test_project_version_is_pep_440_compliant(self):
        try:
            from packaging.version import Version
        except ImportError:
            from pip._vendor.packaging.version import Version
        project=tomllib.loads((Path(__file__).resolve().parents[1]/"pyproject.toml").read_text(encoding="utf-8"))
        version=project["project"]["version"]
        self.assertEqual(str(Version(version)),version)
        self.assertEqual(version,"0.1.0")


if __name__ == "__main__":
    unittest.main()
