"""v0.1F: exact extremality, regime continuity, headless solves and trust policy."""

from dataclasses import replace
from math import sqrt
import unittest

import numpy as np

from kerr_geodesics import (
    EventKind, HorizonRegime, IntegrationSettings, KerrSchildGeometry,
    RayDefinition, ReliabilityLevel, SpacetimeParameters, TerminationStatus,
    assess_trajectory, create_initial_conditions, solve_ray_for_spin,
    solve_rays, sweep_ray_over_spins,
)


SETTINGS = IntegrationSettings(escape_radius=8.0, max_affine_parameter=20.0)
SPINS = (.95, .99, .999, .9999, 1.0, 1.0001, 1.001, 1.01, 1.05)


def geometry(a):
    return KerrSchildGeometry(SpacetimeParameters(a_star=a))


def good(result):
    assert assess_trajectory(result).level is ReliabilityLevel.COMPLETE, result.message
    assert result.max_scaled_null_residual < 1e-8
    assert result.max_scaled_Lz_drift < 1e-8
    assert result.max_scaled_carter_constant_drift < 1e-8


class ExtremalAndRegimeTests(unittest.TestCase):
    def test_exact_extremal_horizon_has_only_one_geometric_root(self):
        horizon = geometry(1.0).horizons()
        self.assertEqual(horizon.regime, HorizonRegime.EXTREMAL)
        self.assertEqual((horizon.r_plus, horizon.r_minus), (1., 1.))
        self.assertFalse(horizon.has_distinct_inner_horizon)
        cases = (
            ([4., .3, .6], [1., 0., 0.], TerminationStatus.ESCAPED, []),
            ([4., 0., 0.], [-1., 0., 0.], TerminationStatus.SINGULARITY_APPROACH,
             ["INWARD"]),
            ([4., .5, .6], [-1., 0., 0.], TerminationStatus.ESCAPED,
             ["INWARD", "OUTWARD"]),
            ([3., .2, 1.], [-1., 0., 0.], TerminationStatus.ESCAPED,
             ["INWARD", "OUTWARD"]),
            ([.15, .05, 3.], [0., 0., -1.], TerminationStatus.ESCAPED,
             ["INWARD"]),
        )
        for position, direction, fate, directions in cases:
            with self.subTest(position=position, direction=direction):
                result = solve_ray_for_spin(RayDefinition(position, direction), 1., SETTINGS)
                self.assertEqual(result.status, fate, result.message)
                self.assertEqual([event.direction for event in result.horizon_crossings], directions)
                self.assertTrue(all(event.kind == EventKind.OUTER_HORIZON
                                    for event in result.horizon_crossings))
                self.assertNotIn("INNER_HORIZON", [t.kind for t in result.region_transitions])
                self.assertTrue(all(event.radial_coordinate == 1.
                                    or abs(event.radial_coordinate-1.) < 1e-12
                                    for event in result.horizon_crossings))
                good(result)
                if len(directions) == 2:
                    self.assertEqual([t.kind for t in result.region_transitions].count("CHART_HANDOFF"), 1)
                    self.assertIn("EXTREMAL_INTERIOR", result.sample_regions)

    def test_near_extremal_sweep_across_both_sides_preserves_physical_events(self):
        inclined = RayDefinition([4., .5, .6], [-1., 0., 0.])
        equatorial = RayDefinition([4., 0., 0.], [-1., 0., 0.])
        axis_near = RayDefinition([.15, .05, 3.], [0., 0., -1.])
        for spin in SPINS:
            with self.subTest(spin=spin):
                horizon = geometry(spin).horizons()
                expected = (HorizonRegime.SUB_EXTREMAL if spin < 1. else
                            HorizonRegime.EXTREMAL if spin == 1. else
                            HorizonRegime.SUPER_EXTREMAL)
                self.assertEqual(horizon.regime, expected)
                inclined_result, equat_result, axis_result = solve_rays(
                    (inclined, equatorial, axis_near), spin, SETTINGS)
                self.assertEqual(inclined_result.status, TerminationStatus.ESCAPED)
                self.assertEqual(equat_result.status, TerminationStatus.SINGULARITY_APPROACH)
                self.assertEqual(axis_result.status, TerminationStatus.ESCAPED)
                count = 4 if spin < 1. else 2 if spin == 1. else 0
                self.assertEqual(len(inclined_result.horizon_crossings), count)
                self.assertEqual(len(equat_result.horizon_crossings),
                                 2 if spin < 1. else 1 if spin == 1. else 0)
                self.assertEqual(len(axis_result.horizon_crossings),
                                 2 if spin < 1. else 1 if spin == 1. else 0)
                self.assertIn("DISK_CROSSING", [t.kind for t in axis_result.region_transitions])
                self.assertEqual(axis_result.sample_branches[-1], -1)
                for result in (inclined_result, equat_result, axis_result):
                    good(result)

    def test_incompatible_ingoing_inner_horizon_uses_regular_patch(self):
        direction = [-sqrt(.75), .5, 0.]
        result = solve_ray_for_spin(RayDefinition([4., 0., 0.], direction), .95, SETTINGS)
        self.assertEqual(result.status, TerminationStatus.SINGULARITY_APPROACH, result.message)
        events = [event.kind for event in result.horizon_crossings]
        self.assertEqual(events, [EventKind.OUTER_HORIZON, EventKind.INNER_HORIZON])
        handoff = [t for t in result.region_transitions if t.kind == "CHART_HANDOFF"]
        self.assertEqual(len(handoff), 1)
        self.assertLess(result.horizon_crossings[0].affine_parameter,
                        handoff[0].affine_parameter)
        self.assertLess(handoff[0].affine_parameter,
                        result.horizon_crossings[1].affine_parameter)
        good(result)


class SpinSweepAndReliabilityTests(unittest.TestCase):
    def test_new_frame_and_covector_for_each_spin_and_repeated_spin(self):
        ray = RayDefinition([4., .5, .6], [-1., 0., 0.])
        results = sweep_ray_over_spins(ray, (.95, 1., 1.05, .95), SETTINGS)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result.initial.definition is ray for result in results))
        self.assertTrue(all(result.initial.momentum is not results[0].initial.momentum
                            for result in results[1:]))
        np.testing.assert_array_equal(results[0].initial.momentum,
                                      results[-1].initial.momentum)
        np.testing.assert_array_equal(results[0].initial.state,
                                      results[-1].initial.state)
        for spin, result in zip((.95, 1., 1.05, .95), results):
            expected = create_initial_conditions(geometry(spin), ray)
            np.testing.assert_allclose(result.initial.momentum, expected.momentum, atol=2e-15, rtol=0)
            self.assertEqual(result.initial.conserved, expected.conserved)
            good(result)
        for key in ("energy", "angular_momentum_z", "carter_constant"):
            values = [getattr(result.initial.conserved, key) for result in results[:3]]
            self.assertGreater(np.ptp(values), 1e-5, key)

    def test_sequential_batch_matches_independent_solves_without_shared_momenta(self):
        rays = (RayDefinition([4., 0., 0.], [1., 0., 0.]),
                RayDefinition([4., .5, .6], [-1., 0., 0.]))
        batch = solve_rays(rays, .9999, SETTINGS)
        for definition, result in zip(rays, batch):
            single = solve_ray_for_spin(definition, .9999, SETTINGS)
            self.assertEqual(result.status, single.status)
            np.testing.assert_array_equal(result.initial.momentum, single.initial.momentum)
            self.assertEqual(result.initial.conserved, single.initial.conserved)
            self.assertAlmostEqual(result.affine_parameter_end, single.affine_parameter_end, delta=2e-12)
            good(result)
        self.assertIsNot(batch[0].initial.momentum, batch[1].initial.momentum)

    def test_nearby_spins_compare_on_shared_affine_time_not_sample_indices(self):
        ray = RayDefinition([4., .5, .6], [-1., 0., 0.])
        solutions = sweep_ray_over_spins(ray, (.9999, 1., 1.0001), SETTINGS)
        # These coordinates share the ingoing chart over [0,1.2]; handoff
        # occurs much later. np.interp is used only for diagnostic comparison.
        affine_grid = np.linspace(0., 1.2, 41)
        sampled = []
        for result in solutions:
            self.assertGreater(result.affine_parameter_end, affine_grid[-1])
            handoffs = [t for t in result.region_transitions if t.kind == "CHART_HANDOFF"]
            self.assertGreater(handoffs[0].affine_parameter, affine_grid[-1])
            sampled.append(np.column_stack([
                np.interp(affine_grid, result.affine_parameters, result.states[:,i])
                for i in (1, 2, 3)
            ]))
            good(result)
        separations = [np.max(np.linalg.norm(sampled[i+1]-sampled[i], axis=1)) for i in (0,1)]
        self.assertLess(max(separations), 5e-4)
        self.assertGreater(min(separations), 1e-7)

    def test_capture_to_escape_at_fixed_local_launch_is_a_reliable_example(self):
        ray = RayDefinition([4., 0., 0.], [-sqrt(.75), .5, 0.])
        capture, scatter = sweep_ray_over_spins(ray, (.95, 1.05), SETTINGS)
        self.assertEqual(capture.status, TerminationStatus.SINGULARITY_APPROACH)
        self.assertEqual(scatter.status, TerminationStatus.ESCAPED)
        self.assertEqual(len(capture.horizon_crossings), 2)
        self.assertEqual(scatter.horizon_crossings, ())
        self.assertGreater(scatter.minimum_sampled_sigma, 1.)
        for result in (capture, scatter):
            good(result)
        tight = IntegrationSettings(escape_radius=8., max_affine_parameter=20., rtol=1e-11)
        for spin, baseline in ((.95, capture), (1.05, scatter)):
            refined = solve_ray_for_spin(ray, spin, tight)
            self.assertEqual(refined.status, baseline.status)
            self.assertLess(abs(refined.affine_parameter_end-baseline.affine_parameter_end), 2e-6)
            good(refined)

    def test_superextremal_axis_disk_close_passage_and_double_crossing(self):
        cases = ((1.0001, .8, 1), (1.001, .8, 1), (1.01, .8, 1),
                 (1.05, 1., 1), (1.2, 1.3, 2))
        for spin, x, n_disk in cases:
            with self.subTest(spin=spin):
                ray = RayDefinition([x, .1, 3.], [0., 0., -1.])
                result = solve_ray_for_spin(ray, spin, SETTINGS)
                self.assertEqual(result.status, TerminationStatus.ESCAPED, result.message)
                self.assertEqual(result.horizon_crossings, ())
                self.assertEqual([t.kind for t in result.region_transitions].count("DISK_CROSSING"), n_disk)
                self.assertLess(result.minimum_sampled_abs_r, 1e-11)
                self.assertGreater(result.minimum_sampled_sigma, .2)
                good(result)

    def test_superextremal_disk_crossings_converge_with_tighter_tolerance(self):
        ray = RayDefinition([1.3, .1, 3.], [0., 0., -1.])
        baseline = solve_ray_for_spin(ray, 1.2, SETTINGS)
        refined = solve_ray_for_spin(ray, 1.2,
                                    IntegrationSettings(escape_radius=8., max_affine_parameter=20., rtol=1e-11))
        self.assertEqual([t.kind for t in baseline.region_transitions].count("DISK_CROSSING"), 2)
        self.assertEqual([t.kind for t in refined.region_transitions].count("DISK_CROSSING"), 2)
        self.assertLess(abs(baseline.affine_parameter_end-refined.affine_parameter_end), 1e-6)
        for one, two in zip((t for t in baseline.region_transitions if t.kind == "DISK_CROSSING"),
                            (t for t in refined.region_transitions if t.kind == "DISK_CROSSING")):
            self.assertLess(abs(one.affine_parameter-two.affine_parameter), 1e-6)
        good(baseline)
        good(refined)

    def test_negative_spin_regime_selected_exterior_solution(self):
        result = solve_ray_for_spin(RayDefinition([4., .1, .6], [1., 0., 0.]), -.999, SETTINGS)
        self.assertEqual(result.status, TerminationStatus.ESCAPED)
        self.assertEqual(result.horizon_crossings, ())
        good(result)

    def test_reliability_distinguishes_fates_partial_and_unreliable_status(self):
        result = solve_ray_for_spin(RayDefinition([4., 0., 0.], [1., 0., 0.]), 0., SETTINGS)
        self.assertEqual(assess_trajectory(result).physical_outcome, TerminationStatus.ESCAPED)
        for status in (TerminationStatus.CHART_BOUNDARY, TerminationStatus.MAX_INTEGRATION_LIMIT):
            assessment = assess_trajectory(replace(result, status=status))
            self.assertEqual(assessment.level, ReliabilityLevel.PARTIAL)
            self.assertIsNone(assessment.physical_outcome)
        for status in (TerminationStatus.NUMERICAL_ERROR, TerminationStatus.DIAGNOSTIC_FAILURE):
            assessment = assess_trajectory(replace(result, status=status))
            self.assertEqual(assessment.level, ReliabilityLevel.UNRELIABLE)
            self.assertIsNone(assessment.physical_outcome)
        tampered = replace(result, max_scaled_carter_constant_drift=1.01e-8)
        self.assertEqual(assess_trajectory(tampered).level, ReliabilityLevel.UNRELIABLE)
        self.assertIsNone(assess_trajectory(tampered).physical_outcome)
        disordered = replace(result, affine_parameters=result.affine_parameters[::-1])
        self.assertEqual(assess_trajectory(disordered).level, ReliabilityLevel.UNRELIABLE)
        bad_chart = replace(result, sample_charts=("invalid",) + result.sample_charts[1:])
        self.assertEqual(assess_trajectory(bad_chart).level, ReliabilityLevel.UNRELIABLE)
        bad_endpoint = result.states.copy()
        bad_endpoint[-1, 1] += .2
        self.assertEqual(assess_trajectory(replace(result, states=bad_endpoint)).level,
                         ReliabilityLevel.UNRELIABLE)

    def test_invalid_inputs_and_empty_batch_are_explicit(self):
        ray = RayDefinition([4., 0., 0.], [1., 0., 0.])
        self.assertEqual(sweep_ray_over_spins(ray, (), SETTINGS), ())
        self.assertEqual(solve_rays((), .99, SETTINGS), ())
        for spin in (float("nan"), float("inf"), True, "1"):
            with self.subTest(spin=spin):
                with self.assertRaises(ValueError):
                    solve_ray_for_spin(ray, spin, SETTINGS)
        with self.assertRaises(TypeError):
            solve_rays((ray, object()), 0., SETTINGS)


if __name__ == "__main__":
    unittest.main()
