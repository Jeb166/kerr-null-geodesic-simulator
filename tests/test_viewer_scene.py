"""No Qt or OpenGL dependency: coordinate, mesh and navigation smoke tests."""

import unittest

import numpy as np

from kerr_geodesics import (
    IntegrationSettings, KerrSchildGeometry, RayDefinition,
    SpacetimeParameters, solve_ray_for_spin,
)
from kerr_viewer.camera import FreeCamera
from kerr_viewer.scene import (
    geometry_scene, horizon_mesh, horizon_visible, ribbon_vertices,
    singularity_marker, trajectory_to_display,
)


class ViewerSceneTests(unittest.TestCase):
    def test_spin_regimes_and_marker_types(self):
        for spin, present, marker_kind in ((0., True, "POINT"), (.7, True, "RING"),
                                           (1., True, "RING"), (1.05, False, "RING"),
                                           (1.2, False, "RING")):
            with self.subTest(spin=spin):
                scene = geometry_scene(spin)
                self.assertEqual(horizon_visible(scene.horizon), present)
                self.assertEqual(scene.singularity.kind, marker_kind)
                vertices = horizon_mesh(scene.horizon, spin, 8, 16)
                self.assertEqual(vertices.shape, (8*16*6, 3) if present else (0, 3))
                self.assertEqual(vertices.dtype, np.float32)
                if present:
                    self.assertAlmostEqual(float(np.max(vertices[:, 2])), scene.horizon.r_plus)
                    self.assertAlmostEqual(float(np.max(np.linalg.norm(vertices[:, :2], axis=1))),
                                           np.hypot(scene.horizon.r_plus, spin), places=6)

    def test_singularity_ring_is_equatorial_and_closed(self):
        for spin in (-1.2, .7, 1.):
            marker = singularity_marker(geometry_scene(spin).singularity)
            self.assertTrue(np.allclose(marker[:, 2], 0))
            self.assertTrue(np.allclose(np.linalg.norm(marker[:, :2], axis=1), abs(spin)))
            self.assertTrue(np.allclose(marker[0], marker[-1]))
            self.assertFalse(marker.flags.writeable)
        self.assertEqual(singularity_marker(geometry_scene(0).singularity).shape, (1, 3))

    def test_ribbon_buffers_are_full_length_and_readonly(self):
        raw = np.array([[3., 0., 1.], [.1, 0., 0.], [.1, 0., -.01], [4., 0., 0.]])
        vertices = ribbon_vertices(raw)
        self.assertEqual(vertices.shape, (18, 8))
        self.assertEqual(vertices.dtype, np.float32)
        self.assertFalse(vertices.flags.writeable)
        np.testing.assert_allclose(vertices[-1, 3:6], raw[-1], atol=1e-7)

    def test_camera_view_projection_move_and_orbit(self):
        camera = FreeCamera()
        origin = camera.view() @ np.r_[camera.position, 1.]
        np.testing.assert_allclose(origin[:3], 0, atol=1e-12)
        center = camera.projection(16/9) @ camera.view() @ [0., 0., 0., 1.]
        np.testing.assert_allclose(center[:2], 0, atol=1e-12)
        camera.move(forward=1, dt=.01)
        self.assertGreater(np.linalg.norm(camera.position - np.array([8., -12., 6.])), 0)
        radius = np.linalg.norm(camera.position)
        camera.orbit(200, 20)
        self.assertAlmostEqual(np.linalg.norm(camera.position), radius, places=12)
        self.assertTrue(np.allclose(camera.position + radius*camera.forward(), 0))

    def test_chart_handoff_coordinates_map_back_exactly(self):
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=1.2), chart="ingoing")
        # Super-extremal transition at 0 < r < 1, away from the ring.
        initial = np.array([0., .5, .1, .8, -1., .2, .1, .1])
        outgoing = geometry.horizon_chart_handoff(initial, "outgoing")
        self.assertIsNotNone(outgoing)
        new_geometry, transformed = outgoing
        np.testing.assert_allclose(new_geometry.canonical_ingoing_position(transformed[1:4]),
                                   initial[1:4], atol=1e-13)

    def test_actual_inner_chart_transition_and_signed_disk(self):
        settings = IntegrationSettings(escape_radius=8., max_affine_parameter=20.)
        ray = RayDefinition([4., .5, .6], [-1., 0., 0.])
        result = solve_ray_for_spin(ray, .7, settings)
        self.assertIn("outgoing", result.sample_charts)
        converted = trajectory_to_display("inner return", result)
        self.assertEqual(len(converted.positions), len(result.states))
        self.assertFalse(converted.positions.flags.writeable)
        handoffs = 0
        for i, (left, right) in enumerate(zip(result.sample_charts[:-1], result.sample_charts[1:])):
            if left != right:
                handoffs += 1
                # The old/new samples have different affine times, so this
                # bounds physical step displacement rather than requiring zero.
                self.assertLess(np.linalg.norm(converted.positions[i+1]-converted.positions[i]), .15)
        self.assertGreaterEqual(handoffs, 1)
        disk = solve_ray_for_spin(RayDefinition([.8, .1, 3.], [0., 0., -1.]),
                                  1.2, settings)
        display = trajectory_to_display("disk", disk)
        self.assertIn(-1, display.branches)
        self.assertIn(1, display.branches)
        self.assertTrue(np.all(np.isfinite(display.positions)))
        self.assertEqual(ribbon_vertices(display.positions).shape[0],
                         6*(len(display.positions)-1))


if __name__ == "__main__":
    unittest.main()
