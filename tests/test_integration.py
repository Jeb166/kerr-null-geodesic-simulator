"""v0.1D real trajectories, event ordering and honest failure classes."""

from math import sqrt
import unittest

import numpy as np

from kerr_geodesics import (
    EventKind,
    IntegrationSettings,
    KerrSchildGeometry,
    RayDefinition,
    SpacetimeParameters,
    TerminationStatus,
    eulerian_frame,
    integrate_ray,
)


def geometry(spin):
    return KerrSchildGeometry(SpacetimeParameters(a_star=spin))


class SingleRayIntegrationTests(unittest.TestCase):
    def test_schwarzschild_radial_outgoing_hits_finite_escape_boundary(self):
        g = geometry(0.0)
        ray = RayDefinition([4., 0., 0.], [1., 0., 0.])
        result = integrate_ray(g, ray, IntegrationSettings(escape_radius=8.0))
        self.assertEqual(result.status, TerminationStatus.ESCAPED)
        self.assertEqual(len(result.horizon_crossings), 0)
        self.assertAlmostEqual(g.radial_coordinate(result.states[-1, 1:4]), 8.0, delta=2e-12)
        # Independent Schwarzschild radial-null equation: dr/dlambda=E.
        self.assertAlmostEqual(result.affine_parameter_end, 4.0 / result.initial.conserved.energy, delta=2e-8)
        self.assertTrue(np.all(np.diff(result.affine_parameters) > 0))
        self.assertEqual(len(result.states), len(result.affine_parameters))
        np.testing.assert_array_equal(result.states[:, 4], np.full(len(result.states), result.initial.momentum[0]))
        self.assertLess(result.max_scaled_null_residual, 1e-9)

    def test_schwarzschild_radial_inward_crosses_horizon_then_approaches_point(self):
        g = geometry(0.0)
        settings = IntegrationSettings(escape_radius=8.0, epsilon_sigma=1e-4)
        result = integrate_ray(g, RayDefinition([4., 0., 0.], [-1., 0., 0.]), settings)
        self.assertEqual(result.status, TerminationStatus.SINGULARITY_APPROACH)
        self.assertEqual([h.kind for h in result.horizon_crossings], [EventKind.OUTER_HORIZON])
        self.assertEqual(result.horizon_crossings[0].direction, "INWARD")
        self.assertAlmostEqual(result.horizon_crossings[0].radial_coordinate, 2.0, delta=2e-12)
        self.assertGreater(result.affine_parameter_end, result.horizon_crossings[0].affine_parameter)
        self.assertAlmostEqual(result.minimum_sampled_sigma, 1e-4, delta=2e-14)
        self.assertAlmostEqual(result.minimum_sampled_r, 1e-2, delta=2e-12)
        self.assertLess(result.max_scaled_null_residual, 1e-9)
        self.assertEqual(result.max_abs_Lz_drift, 0.0)

    def test_schwarzschild_circular_photon_orbit_at_r3_and_critical_b(self):
        g = geometry(0.0)
        position = np.array([3., 0., 0.])
        # Analytic KS tangent for circular photon orbit: k=(3,0,sqrt(3),0).
        # It is null, dr/dlambda=0 and L_z/E = 3sqrt(3).
        known_tangent = np.array([3., 0., sqrt(3), 0.])
        known_covector = g.metric(position) @ known_tangent
        frame = eulerian_frame(g, position)
        local_energy = -(known_covector @ frame.observer)
        local_direction = (frame.triad @ known_covector) / local_energy
        self.assertAlmostEqual(np.linalg.norm(local_direction), 1.0, delta=1e-14)
        result = integrate_ray(
            g, RayDefinition(position, local_direction),
            IntegrationSettings(max_affine_parameter=2.0, escape_radius=8.0, max_step=0.1),
        )
        self.assertEqual(result.status, TerminationStatus.MAX_INTEGRATION_LIMIT)
        self.assertAlmostEqual(
            result.initial.conserved.angular_momentum_z / result.initial.conserved.energy,
            3 * sqrt(3), delta=1e-12,
        )
        self.assertLess(max(abs(g.radial_coordinate(s[1:4]) - 3.0) for s in result.states), 2e-8)
        self.assertLess(result.max_scaled_null_residual, 1e-9)

    def test_subextremal_kerr_outer_and_inner_events_do_not_terminate(self):
        g = geometry(0.7)
        result = integrate_ray(g, RayDefinition([4., 0., 0.], [-1., 0., 0.]),
                               IntegrationSettings(escape_radius=8.0))
        self.assertEqual(result.status, TerminationStatus.SINGULARITY_APPROACH)
        self.assertEqual([h.kind for h in result.horizon_crossings],
                         [EventKind.OUTER_HORIZON, EventKind.INNER_HORIZON])
        horizon = g.horizons()
        for event, expected in zip(result.horizon_crossings, (horizon.r_plus, horizon.r_minus)):
            self.assertEqual(event.direction, "INWARD")
            self.assertAlmostEqual(event.radial_coordinate, expected, delta=1e-10)
            self.assertTrue(np.all(np.isfinite(event.state)))
            self.assertLess(event.affine_parameter, result.affine_parameter_end)
        self.assertLess(result.max_scaled_null_residual, 1e-8)
        self.assertLess(result.max_scaled_Lz_drift, 1e-8)

    def test_regular_disk_crossing_continues_on_negative_sheet(self):
        g = geometry(0.8)
        result = integrate_ray(g, RayDefinition([0., 0., 3.], [0., 0., -1.]),
                               IntegrationSettings(escape_radius=8.0))
        self.assertEqual(result.status, TerminationStatus.ESCAPED)
        self.assertLess(result.minimum_sampled_r, -7.9)
        self.assertLess(result.minimum_sampled_abs_r, 1e-12)
        self.assertGreater(result.minimum_sampled_sigma, 0.6)
        self.assertEqual([h.kind for h in result.horizon_crossings],
                         [EventKind.OUTER_HORIZON, EventKind.INNER_HORIZON])
        self.assertLess(result.max_scaled_null_residual, 1e-9)
        self.assertEqual([t.kind for t in result.region_transitions].count("DISK_CROSSING"), 1)
        self.assertEqual(result.sample_regions[-1], "NEGATIVE_R")

    def test_near_extremal_and_super_extremal_exterior_smoke(self):
        ray = RayDefinition([4., 0.1, 0.5], [1., 0., 0.])
        for spin in (0.999, 1.05):
            with self.subTest(spin=spin):
                g = geometry(spin)
                result = integrate_ray(g, ray, IntegrationSettings(escape_radius=8.0))
                self.assertEqual(result.status, TerminationStatus.ESCAPED)
                self.assertEqual(result.horizon_crossings, ())
                self.assertLess(result.max_scaled_null_residual, 1e-8)
                self.assertLess(result.max_scaled_Lz_drift, 1e-8)
                self.assertLess(result.max_scaled_carter_constant_drift, 1e-8)
                if spin > 1:
                    self.assertFalse(g.horizons().exists)

    def test_nonzero_carter_constant_stays_stable_in_exterior(self):
        g = geometry(0.7)
        direction = np.array([.7, .2, .35]) / np.linalg.norm([.7, .2, .35])
        result = integrate_ray(g, RayDefinition([4., 1., .6], direction),
                               IntegrationSettings(escape_radius=9.0))
        self.assertEqual(result.status, TerminationStatus.ESCAPED)
        self.assertGreater(result.initial.conserved.carter_constant, 1.0)
        self.assertLess(result.max_scaled_carter_constant_drift, 1e-9)
        self.assertLess(result.max_scaled_Lz_drift, 1e-9)

    def test_escape_requires_outward_crossing(self):
        g = geometry(0.0)
        # Starts outside r_escape and crosses r=5 inward, never outward.
        result = integrate_ray(g, RayDefinition([6., 0., 0.], [-1., 0., 0.]),
                               IntegrationSettings(escape_radius=5.0, max_affine_parameter=2.0))
        self.assertEqual(result.status, TerminationStatus.MAX_INTEGRATION_LIMIT)
        self.assertLess(result.minimum_sampled_r, 5.0)

    def test_step_and_affine_budgets_are_distinct_from_escape(self):
        g = geometry(0.0)
        ray = RayDefinition([4., 0., 0.], [1., 0., 0.])
        steps = integrate_ray(g, ray, IntegrationSettings(escape_radius=8.0, max_accepted_steps=2))
        self.assertEqual(steps.status, TerminationStatus.MAX_INTEGRATION_LIMIT)
        self.assertEqual(steps.accepted_steps, 2)
        time = integrate_ray(g, ray, IntegrationSettings(escape_radius=8.0, max_affine_parameter=0.1))
        self.assertEqual(time.status, TerminationStatus.MAX_INTEGRATION_LIMIT)
        self.assertAlmostEqual(time.affine_parameter_end, 0.1, delta=1e-14)

    def test_schwarzschild_approach_threshold_converges_toward_r0(self):
        g = geometry(0.0)
        ray = RayDefinition([4., 0., 0.], [-1., 0., 0.])
        coarse = integrate_ray(g, ray, IntegrationSettings(escape_radius=8., epsilon_sigma=1e-4))
        fine = integrate_ray(g, ray, IntegrationSettings(escape_radius=8., epsilon_sigma=2.5e-5))
        self.assertEqual(coarse.status, TerminationStatus.SINGULARITY_APPROACH)
        self.assertEqual(fine.status, TerminationStatus.SINGULARITY_APPROACH)
        self.assertAlmostEqual(coarse.minimum_sampled_r, .01, delta=2e-12)
        self.assertAlmostEqual(fine.minimum_sampled_r, .005, delta=2e-12)
        self.assertGreater(fine.affine_parameter_end, coarse.affine_parameter_end)
        self.assertLess(abs(coarse.horizon_crossings[0].affine_parameter
                            -fine.horizon_crossings[0].affine_parameter), 1e-9)

    def test_solver_failure_reports_numerical_error_with_message(self):
        class FailingGeometry(KerrSchildGeometry):
            def inverse_metric_derivatives(self, position):
                raise RuntimeError("deliberate derivative failure")

        g = FailingGeometry(SpacetimeParameters(a_star=0.0))
        result = integrate_ray(g, RayDefinition([4., 0., 0.], [1., 0., 0.]))
        self.assertEqual(result.status, TerminationStatus.NUMERICAL_ERROR)
        self.assertIn("deliberate derivative failure", result.message)
        self.assertEqual(result.accepted_steps, 0)

    def test_tolerance_convergence_of_outgoing_and_horizon_crossing_rays(self):
        for spin, position, direction in (
            (0.0, [4., 0., 0.], [1., 0., 0.]),
            (0.7, [4., 0., 0.], [-1., 0., 0.]),
        ):
            with self.subTest(spin=spin):
                g, ray = geometry(spin), RayDefinition(position, direction)
                common = dict(escape_radius=8.0)
                loose = integrate_ray(g, ray, IntegrationSettings(rtol=1e-9, **common))
                tight = integrate_ray(g, ray, IntegrationSettings(rtol=1e-11, **common))
                self.assertEqual(loose.status, tight.status)
                self.assertLess(abs(loose.affine_parameter_end-tight.affine_parameter_end), 1e-8)
                self.assertLess(np.max(np.abs(loose.states[-1]-tight.states[-1])), 5e-6)
                self.assertEqual(len(loose.horizon_crossings), len(tight.horizon_crossings))
                for one, two in zip(loose.horizon_crossings, tight.horizon_crossings):
                    self.assertEqual(one.kind, two.kind)
                    self.assertLess(abs(one.affine_parameter-two.affine_parameter), 1e-8)
                self.assertLess(tight.max_scaled_null_residual, 1e-8)
                self.assertLess(tight.max_scaled_Lz_drift, 1e-8)


if __name__ == "__main__":
    unittest.main()
