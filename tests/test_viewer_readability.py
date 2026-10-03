"""Headless presentation checks; existing physics trajectories remain intact."""

from dataclasses import replace
import unittest

import numpy as np

from kerr_geodesics import IntegrationSettings, RayDefinition, solve_ray_for_spin
from kerr_viewer.scene import DisplayEvent, DisplayRay, trajectory_to_display
from kerr_viewer.renderer import _arrow_vertices
from kerr_viewer.visuals import (
    direction_arrows, endpoint_glyph, ray_color, ray_markers,
    ray_opacity, termination_reason,
)


def display(status="ESCAPED", reliability="COMPLETE"):
    positions = np.array([[3., 1., 0.], [2., 1., 0.], [1., 1., 0.]])
    positions.setflags(write=False)
    return DisplayRay("Example", positions, status, reliability,
                      ("ingoing",)*3, (1,)*3, (), 8.)


class RayReadabilityTests(unittest.TestCase):
    def test_palette_is_stable_and_distinct(self):
        palette = [ray_color(i) for i in range(3)]
        self.assertEqual([c.name for c in palette], ["Yellow", "Cyan", "Magenta"])
        self.assertEqual(len(set(c.hex for c in palette)), 3)
        self.assertEqual(ray_color(0), ray_color(3))

    def test_start_and_end_are_exact_physics_samples(self):
        ray = display()
        start, end = ray_markers(ray, 1)
        self.assertEqual(start.glyph, 1)
        self.assertEqual(end.glyph, 2)
        np.testing.assert_array_equal(start.position, ray.positions[0])
        np.testing.assert_array_equal(end.position, ray.positions[-1])
        self.assertEqual(start.rgba[:3], ray_color(1).rgb)

    def test_fates_partial_and_unreliable_have_distinct_glyphs(self):
        names = ["ESCAPED", "SINGULARITY_APPROACH", "MAX_INTEGRATION_LIMIT",
                 "DIAGNOSTIC_FAILURE", "NUMERICAL_ERROR"]
        self.assertEqual([endpoint_glyph(name) for name in names], [2, 3, 4, 5, 6])
        self.assertEqual(ray_opacity("COMPLETE"), 1.0)
        self.assertLess(ray_opacity("UNRELIABLE"), ray_opacity("PARTIAL"))
        self.assertLess(ray_opacity("PARTIAL"), 1.0)
        self.assertEqual(ray_markers(display("DIAGNOSTIC_FAILURE", "UNRELIABLE"), 0)[1].glyph, 5)
        self.assertEqual(ray_markers(display("MAX_INTEGRATION_LIMIT", "PARTIAL"), 0)[1].glyph, 4)
        self.assertEqual(len(ray_markers(display("CANCELLED", "UNRELIABLE"), 0)), 1)
        self.assertEqual(len(ray_markers(display("SUPERSEDED", "UNRELIABLE"), 0)), 1)

    def test_finite_escape_and_sigma_reason(self):
        self.assertIn("finite escape boundary", termination_reason("ESCAPED", 8.))
        self.assertIn("R_escape=8", termination_reason("ESCAPED", 8.))
        self.assertIn("Sigma threshold", termination_reason("SINGULARITY_APPROACH"))
        self.assertIn("undetermined", termination_reason("MAX_INTEGRATION_LIMIT"))

    def test_arrows_aligned_with_sample_order_and_sparse(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [4., 0., 0.]])
        arrows = direction_arrows(points)
        self.assertEqual(len(arrows), 4)
        for arrow in arrows:
            tangent = np.asarray(arrow.end)-np.asarray(arrow.start)
            self.assertGreater(tangent[0], 0.)
            self.assertGreaterEqual(arrow.fraction, 0.)
            self.assertLessEqual(arrow.fraction, 1.)
        triangles = _arrow_vertices(points)
        self.assertEqual(triangles.shape, (48, 9))
        # Arrow arm tails are behind the tip in shader-local tangent pixels.
        self.assertLess(float(np.min(triangles[:, -2])), -10.)
        self.assertGreater(float(np.max(triangles[:, -2])), -2.)

    def test_event_marker_uses_given_location_not_nearest_sample(self):
        marker_position = np.array([1.45, .2, .3])
        event = DisplayEvent("DISK_CROSSING", marker_position, "DISK", .4)
        ray = replace(display(), events=(event,))
        markers = ray_markers(ray, 2)
        self.assertEqual(len(markers), 3)
        self.assertEqual(markers[2].glyph, 8)
        np.testing.assert_array_equal(markers[2].position, marker_position)

    def test_recorded_horizon_state_and_disk_chart_lambdas(self):
        settings = IntegrationSettings(escape_radius=8., max_affine_parameter=20.)
        result = solve_ray_for_spin(RayDefinition([4., .5, .6], [-1., 0., 0.]),
                                    .7, settings)
        positions_original = result.states.copy()
        display_ray = trajectory_to_display("return", result)
        horizon_events = [event for event in display_ray.events if "HORIZON" in event.kind]
        self.assertEqual(len(horizon_events), len(result.horizon_crossings))
        self.assertEqual([e.label for e in horizon_events],
                         ["OUTER-I", "INNER-I", "INNER-O", "OUTER-O"])
        for event in display_ray.events:
            if event.kind == "CHART_HANDOFF":
                matches = np.where(abs(result.affine_parameters-event.affine_parameter) < 1e-10)[0]
                self.assertGreater(len(matches), 0)
                np.testing.assert_allclose(event.position, display_ray.positions[matches[0]], atol=1e-12)
        np.testing.assert_array_equal(result.states, positions_original)

        disk_result = solve_ray_for_spin(RayDefinition([.8, .1, 3.], [0., 0., -1.]),
                                         1.2, settings)
        disk = trajectory_to_display("disk", disk_result)
        crossings = [event for event in disk.events if event.kind == "DISK_CROSSING"]
        self.assertGreaterEqual(len(crossings), 1)
        for event in crossings:
            matches = np.where(abs(disk_result.affine_parameters-event.affine_parameter) < 1e-10)[0]
            self.assertGreater(len(matches), 0)
            np.testing.assert_allclose(event.position, disk.positions[matches[0]], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
