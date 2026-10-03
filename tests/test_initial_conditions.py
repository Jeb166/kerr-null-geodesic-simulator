"""v0.1C local observer, symmetric triad and null launch tests."""

import math
import unittest

import numpy as np

from kerr_geodesics import (
    CoordinateDomainError,
    FRAME_CONVENTION,
    KerrSchildGeometry,
    RayDefinition,
    SpacetimeParameters,
    create_initial_conditions,
    eulerian_frame,
    hamiltonian_value,
)

from frame_samples import DIRECTIONS, SPINS, positions_for


class FrameAndLaunchTests(unittest.TestCase):
    def test_frame_orthonormality_and_future_normal_on_both_horizon_sides(self):
        for spin in SPINS:
            geometry = KerrSchildGeometry(SpacetimeParameters(a_star=spin))
            for position in positions_for(geometry):
                with self.subTest(spin=spin, position=position):
                    g = geometry.metric(position)
                    frame = eulerian_frame(geometry, position)
                    n, triad = frame.observer, frame.triad
                    self.assertGreater(n[0], 0.0)
                    self.assertLess(abs(n @ g @ n + 1), 3e-12)
                    np.testing.assert_allclose(triad @ g @ triad.T, np.eye(3), rtol=0, atol=3e-12)
                    np.testing.assert_allclose(triad @ g @ n, np.zeros(3), rtol=0, atol=3e-12)
                    np.testing.assert_allclose(g @ n, [-frame.lapse, 0, 0, 0], rtol=0, atol=3e-12)
                    np.testing.assert_array_equal(triad[:, 0], np.zeros(3))

    def test_null_tangent_momentum_hamiltonian_and_local_energy(self):
        for spin in SPINS:
            geometry = KerrSchildGeometry(SpacetimeParameters(a_star=spin))
            for position in positions_for(geometry):
                for direction in DIRECTIONS:
                    with self.subTest(spin=spin, position=position, direction=direction):
                        definition = RayDefinition(position, direction)
                        result = create_initial_conditions(geometry, definition)
                        g = geometry.metric(position)
                        inv = geometry.inverse_metric(position)
                        self.assertEqual(result.state.shape, (8,))
                        self.assertGreater(result.tangent[0], 0.0)
                        self.assertLess(abs(result.tangent @ g @ result.tangent), 3e-12)
                        self.assertLess(abs(result.momentum @ inv @ result.momentum), 3e-12)
                        self.assertLess(abs(hamiltonian_value(geometry, result.state)), 2e-12)
                        self.assertAlmostEqual(result.local_energy, 1.0, delta=3e-12)
                        np.testing.assert_allclose(inv @ result.momentum, result.tangent, rtol=0, atol=3e-12)
                        np.testing.assert_allclose(
                            result.frame.triad @ result.momentum,
                            direction, rtol=0, atol=3e-12,
                        )

    def test_carter_constant_against_separated_formula_and_pole(self):
        for spin in SPINS:
            geometry = KerrSchildGeometry(SpacetimeParameters(a_star=spin))
            position = np.array([2.4, -1.0, 0.9])
            result = create_initial_conditions(
                geometry, RayDefinition(position, DIRECTIONS[-1])
            )
            r = geometry.radial_coordinate(position)
            cosine = position[2] / r
            sine = math.hypot(position[0], position[1]) / math.hypot(r, spin)
            p = result.momentum
            p_theta = (cosine / sine) * (position[0] * p[1] + position[1] * p[2]) - r * sine * p[3]
            reference = p_theta**2 + cosine**2 * (
                result.conserved.angular_momentum_z**2 / sine**2
                - spin**2 * result.conserved.energy**2
            )
            self.assertAlmostEqual(result.conserved.carter_constant, reference, delta=2e-12)
            self.assertAlmostEqual(result.conserved.energy, -p[0], delta=1e-15)
            self.assertAlmostEqual(
                result.conserved.angular_momentum_z, position[0]*p[2] - position[1]*p[1],
                delta=1e-15,
            )

            axis = np.array([0.0, 0.0, 2.5])
            axial = create_initial_conditions(geometry, RayDefinition(axis, DIRECTIONS[-1]))
            expected_axis = (2.5**2 + spin**2) * np.sum(axial.momentum[1:3]**2) - spin**2 * axial.conserved.energy**2
            self.assertAlmostEqual(axial.conserved.carter_constant, expected_axis, delta=2e-12)

    def test_schwarzschild_carter_is_L_squared_minus_Lz_squared(self):
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=0))
        for position in positions_for(geometry):
            if np.linalg.norm(position) < 1.0:
                continue
            result = create_initial_conditions(geometry, RayDefinition(position, DIRECTIONS[-1]))
            L = np.cross(position, result.momentum[1:])
            self.assertAlmostEqual(
                result.conserved.carter_constant,
                L @ L - L[2]**2,
                delta=2e-12,
            )

    def test_spin_recompute_uses_same_definition_and_changes_derived_momentum(self):
        ray = RayDefinition(np.array([3., 0.8, 1.1]), DIRECTIONS[-1])
        first = create_initial_conditions(KerrSchildGeometry(SpacetimeParameters(a_star=0.2)), ray)
        second = create_initial_conditions(KerrSchildGeometry(SpacetimeParameters(a_star=0.8)), ray)
        self.assertIs(first.definition, second.definition)
        self.assertEqual(first.definition.frame_convention, FRAME_CONVENTION)
        np.testing.assert_array_equal(first.state[1:4], second.state[1:4])
        self.assertGreater(np.linalg.norm(first.momentum - second.momentum), 1e-3)
        self.assertNotAlmostEqual(first.conserved.carter_constant, second.conserved.carter_constant, delta=1e-5)

    def test_symmetric_triad_is_continuous_through_spin_zero_and_polar_axis(self):
        position = np.array([0., 0., 2.0])
        at_zero = eulerian_frame(KerrSchildGeometry(SpacetimeParameters(a_star=0)), position)
        plus = eulerian_frame(KerrSchildGeometry(SpacetimeParameters(a_star=1e-7)), position)
        minus = eulerian_frame(KerrSchildGeometry(SpacetimeParameters(a_star=-1e-7)), position)
        self.assertLess(np.max(np.abs(plus.triad-at_zero.triad)), 1e-7)
        self.assertLess(np.max(np.abs(minus.triad-at_zero.triad)), 1e-7)
        self.assertLess(np.max(np.abs(plus.triad-minus.triad)), 1e-7)

    def test_invalid_definition_and_coordinate_boundary_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "unit"):
            RayDefinition([3, 0, 0], [2, 0, 0])
        with self.assertRaisesRegex(ValueError, "omega_local=1"):
            RayDefinition([3, 0, 0], [1, 0, 0], omega_local=2)
        with self.assertRaisesRegex(ValueError, "frame_convention"):
            RayDefinition([3, 0, 0], [1, 0, 0], frame_convention="arbitrary")
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=0.8))
        with self.assertRaises(CoordinateDomainError):
            create_initial_conditions(geometry, RayDefinition([0.4, 0, 0], [1, 0, 0]))


if __name__ == "__main__":
    unittest.main()
