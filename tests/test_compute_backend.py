"""v0.2A reference/compiled physics equivalence and compute-control tests."""

from math import sqrt
from threading import Event
import unittest

import numpy as np

from kerr_geodesics import (
    ComputeStatus, EventKind, IntegrationSettings, KerrSchildGeometry,
    LatestWinsScheduler, RayBatchExecutor, RayDefinition, RayEntry,
    ReliabilityLevel, SpacetimeParameters, assess_trajectory,
    create_initial_conditions, eulerian_frame, integrate_ray,
    null_geodesic_rhs, solve_ray_for_spin, solve_rays, warm_compute_backend,
)
from kerr_geodesics._native import HAS_NUMBA, fused_rhs
from kerr_geodesics.integration import ComputationCancelled


SETTINGS = IntegrationSettings(escape_radius=8., max_affine_parameter=20.)


def fixture_rays():
    return (
        ("Schwarzschild outward", 0., RayDefinition([4., 0., 0.], [1., 0., 0.])),
        ("Schwarzschild inward", 0., RayDefinition([4., 0., 0.], [-1., 0., 0.])),
        ("Kerr horizons", .7, RayDefinition([4., 0., 0.], [-1., 0., 0.])),
        ("signed disk", .8, RayDefinition([.4, .1, 3.], [0., 0., -1.])),
        ("extremal", 1., RayDefinition([4., .5, .6], [-1., 0., 0.])),
        ("near extremal", .9999, RayDefinition([4., .5, .6], [-1., 0., 0.])),
        ("capture", .95, RayDefinition([4., 0., 0.], [-sqrt(.75), .5, 0.])),
        ("scatter", 1.05, RayDefinition([4., 0., 0.], [-sqrt(.75), .5, 0.])),
        ("double disk", 1.2, RayDefinition([1.3, .1, 3.], [0., 0., -1.])),
    )


class CompiledRhsTests(unittest.TestCase):
    @unittest.skipUnless(HAS_NUMBA, "install the Numba performance extra")
    def test_analytic_fused_rhs_matches_reference_on_both_charts_and_sheets(self):
        warm_compute_backend()
        for spin, branch, chart, hemisphere, position in (
            (.7, 1, "ingoing", 0, (4., .5, .6)),
            (.9999, 1, "outgoing", 0, (.12, .08, .22)),
            (1.2, -1, "ingoing", 0, (.3, .1, -.5)),
            (.8, 1, "ingoing", 1, (.3, .1, .01)),
            (-.8, -1, "outgoing", 0, (.5, .1, .2)),
            (0., 1, "ingoing", 0, (4., 0., .5)),
        ):
            with self.subTest(spin=spin, branch=branch, chart=chart):
                geom = KerrSchildGeometry(SpacetimeParameters(a_star=spin),
                                          branch=branch, chart=chart,
                                          disk_hemisphere=hemisphere or None)
                # An off-shell state verifies the algebra without launch-frame
                # restrictions on some negative-sheet/outgoing slices.
                state = np.array([0., *position, -1., .1, .2, .3])
                expected = null_geodesic_rhs(geom, state)
                actual = fused_rhs(*state[1:4], *state[4:8], spin,
                                   branch, 1 if chart == "ingoing" else -1, hemisphere)
                np.testing.assert_allclose(actual, expected, atol=5e-14, rtol=2e-13)
                self.assertEqual(actual[4], 0.)

    @unittest.skipUnless(HAS_NUMBA, "install the Numba performance extra")
    def test_reference_and_compiled_trajectories_share_status_events_and_diagnostics(self):
        for label, spin, ray in fixture_rays():
            with self.subTest(label=label):
                reference = solve_ray_for_spin(ray, spin, SETTINGS, backend="reference")
                optimized = solve_ray_for_spin(ray, spin, SETTINGS, backend="compiled")
                self.assertEqual(optimized.status, reference.status)
                self.assertEqual(assess_trajectory(optimized).level,
                                 assess_trajectory(reference).level)
                self.assertEqual([(e.kind, e.direction) for e in optimized.horizon_crossings],
                                 [(e.kind, e.direction) for e in reference.horizon_crossings])
                self.assertEqual([(t.kind, t.from_chart, t.to_chart)
                                  for t in optimized.region_transitions],
                                 [(t.kind, t.from_chart, t.to_chart)
                                  for t in reference.region_transitions])
                self.assertEqual(optimized.sample_branches[-1], reference.sample_branches[-1])
                self.assertEqual(optimized.sample_charts[-1], reference.sample_charts[-1])
                self.assertLess(abs(optimized.affine_parameter_end-reference.affine_parameter_end), 2e-6)
                self.assertLess(np.max(abs(optimized.states[-1, 1:4]-reference.states[-1, 1:4])), 2e-5)
                for attr in ("max_scaled_null_residual", "max_scaled_Lz_drift",
                             "max_scaled_carter_constant_drift"):
                    self.assertLess(getattr(optimized, attr), 1e-8)
                    self.assertLess(abs(getattr(optimized, attr)-getattr(reference, attr)), 2e-9)
                for first, second in zip(optimized.horizon_crossings,
                                         reference.horizon_crossings):
                    self.assertLess(abs(first.affine_parameter-second.affine_parameter), 2e-6)
                for first, second in zip(optimized.region_transitions,
                                         reference.region_transitions):
                    self.assertLess(abs(first.affine_parameter-second.affine_parameter), 2e-6)
                # A common affine prefix avoids comparing different adaptive
                # step indices, or states on opposite sides of a patch jump.
                upper = min(.5, reference.affine_parameter_end,
                            optimized.affine_parameter_end)
                grid = np.linspace(0., upper, 16)
                first = np.array([np.interp(grid, reference.affine_parameters,
                                            reference.states[:,axis]) for axis in (1,2,3)])
                second = np.array([np.interp(grid, optimized.affine_parameters,
                                             optimized.states[:,axis]) for axis in (1,2,3)])
                self.assertLess(np.max(abs(first-second)), 5e-5)

    @unittest.skipUnless(HAS_NUMBA, "install the Numba performance extra")
    def test_schwarzschild_photon_sphere_matches_reference(self):
        g = KerrSchildGeometry(SpacetimeParameters(a_star=0.))
        pos = np.array([3., 0., 0.])
        tangent = np.array([3., 0., sqrt(3.), 0.])
        p = g.metric(pos) @ tangent
        frame = eulerian_frame(g, pos)
        local = (frame.triad @ p)/(-p @ frame.observer)
        ray = RayDefinition(pos, local)
        short = IntegrationSettings(max_affine_parameter=2., escape_radius=8., max_step=.1)
        two = [solve_ray_for_spin(ray, 0., short, backend=backend)
               for backend in ("reference", "compiled")]
        self.assertEqual(two[0].status, two[1].status)
        self.assertEqual(assess_trajectory(two[0]).level, ReliabilityLevel.PARTIAL)
        self.assertEqual(assess_trajectory(two[1]).level, ReliabilityLevel.PARTIAL)
        self.assertLess(max(abs(g.radial_coordinate(s[1:4])-3.)
                            for s in two[1].states), 2e-8)
        self.assertLess(abs(two[1].affine_parameter_end-two[0].affine_parameter_end), 1e-12)

    def test_cancel_signal_is_not_a_geodesic_termination_status(self):
        g = KerrSchildGeometry(SpacetimeParameters(a_star=.7), backend="reference")
        ray = RayDefinition([4., .5, .6], [-1., 0., 0.])
        count = 0
        def cancel():
            nonlocal count
            count += 1
            return count > 25
        with self.assertRaises(ComputationCancelled):
            integrate_ray(g, ray, SETTINGS, cancel_requested=cancel)


class BatchAndSchedulerTests(unittest.TestCase):
    @unittest.skipUnless(HAS_NUMBA, "install the Numba performance extra")
    def test_persistent_process_pool_matches_deterministic_sequential_order(self):
        rays = tuple(item[2] for item in fixture_rays()[:5])
        with RayBatchExecutor(2, backend="compiled") as pool:
            for spin in (.9999, 1.2):
                with self.subTest(spin=spin):
                    parallel = pool.solve_rays(rays, spin, SETTINGS)
                    sequential = solve_rays(rays, spin, SETTINGS, backend="compiled")
                    self.assertEqual([r.status for r in parallel],
                                     [r.status for r in sequential])
                    for got, expected in zip(parallel, sequential):
                        self.assertEqual(assess_trajectory(got).level,
                                         assess_trajectory(expected).level)
                        self.assertEqual(got.initial.definition.coordinate_position.tolist(),
                                         expected.initial.definition.coordinate_position.tolist())
                        self.assertEqual([(e.kind,e.direction) for e in got.horizon_crossings],
                                         [(e.kind,e.direction) for e in expected.horizon_crossings])
                        self.assertAlmostEqual(got.affine_parameter_end,
                                               expected.affine_parameter_end, delta=2e-7)

    def test_latest_request_supersedes_running_and_drops_pending(self):
        started = Event()
        def controlled_solver(ray, spin, settings, *, backend, cancel_requested):
            if spin == .95:
                started.set()
                while not cancel_requested():
                    Event().wait(.001)
                raise ComputationCancelled("cooperatively superseded")
            return solve_ray_for_spin(ray, spin, settings, backend="reference",
                                      cancel_requested=cancel_requested)

        ray = RayDefinition([4., 0., 0.], [1., 0., 0.])
        with LatestWinsScheduler(backend="reference", solver=controlled_solver) as scheduler:
            old = scheduler.submit(.95, (RayEntry("main", ray),), SETTINGS)
            self.assertTrue(started.wait(timeout=3.))
            pending = scheduler.submit(.96, (ray,), SETTINGS)
            latest = scheduler.submit(.97, (RayEntry("main", ray),), SETTINGS)
            self.assertLess(old.request_id, pending.request_id)
            self.assertLess(pending.request_id, latest.request_id)
            self.assertEqual(scheduler.wait_for(pending.request_id, timeout=5.).status,
                             ComputeStatus.SUPERSEDED)
            previous = scheduler.wait_for(old.request_id, timeout=5.)
            self.assertEqual(previous.status, ComputeStatus.SUPERSEDED)
            self.assertFalse(previous.publishable)
            self.assertEqual(previous.rays, ())
            final = scheduler.wait_for(latest.request_id, timeout=5.)
            self.assertEqual(final.status, ComputeStatus.DONE)
            self.assertEqual(final.rays[0].ray_id, "main")
            self.assertEqual(scheduler.latest(), final)
            self.assertEqual(final.rays[0].trajectory.status.value, "ESCAPED")
            self.assertFalse(final.rays[0].trajectory.states.flags.writeable)
            self.assertFalse(final.rays[0].trajectory.initial.momentum.flags.writeable)

    def test_request_snapshots_deep_copy_ray_definitions_and_enforce_ids(self):
        ray = RayDefinition([4., 0., 0.], [1., 0., 0.])
        with LatestWinsScheduler(backend="reference") as scheduler:
            with self.assertRaises(ValueError):
                scheduler.submit(1., (RayEntry("same", ray), RayEntry("same", ray)))
            request = scheduler.submit(1., (ray,), SETTINGS)
            self.assertEqual(request.rays[0].ray_id, "ray-0000")
            self.assertIsNot(request.rays[0].definition.coordinate_position,
                             ray.coordinate_position)
            self.assertFalse(request.rays[0].definition.coordinate_position.flags.writeable)
            self.assertEqual(scheduler.wait_for(request.request_id, timeout=5.).status,
                             ComputeStatus.DONE)


if __name__ == "__main__":
    unittest.main()
