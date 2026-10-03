"""v0.1B: pointwise analytic Kerr derivatives and Hamiltonian RHS."""

import math
import unittest

import numpy as np

from kerr_geodesics import (
    CoordinateDomainError,
    Geometry,
    KerrSchildGeometry,
    SpacetimeParameters,
    hamiltonian_value,
    null_geodesic_rhs,
)

from reference_calculations import (
    POSITIONS,
    SPATIAL_MOMENTA,
    SPINS,
    controlled_null_state,
    finite_difference_hamiltonian_gradient,
    finite_difference_inverse_metric,
)


class DerivativeTests(unittest.TestCase):
    def test_analytic_inverse_metric_derivatives_against_central_differences(self):
        for spin in SPINS:
            geometry = KerrSchildGeometry(SpacetimeParameters(a_star=spin))
            for position in POSITIONS:
                with self.subTest(spin=spin, position=position):
                    analytic = geometry.inverse_metric_derivatives(position)
                    reference = finite_difference_inverse_metric(geometry, position)
                    self.assertEqual(analytic.shape, (4, 4, 4))
                    self.assertTrue(np.all(analytic[0] == 0.0))
                    self.assertTrue(np.all(np.isfinite(analytic)))
                    self.assertLess(np.max(np.abs(analytic - reference)), 2e-8)
                    self.assertLess(np.max(np.abs(analytic - analytic.transpose(0, 2, 1))), 1e-14)

    def test_derivatives_finite_at_horizons_and_domain_error_is_preserved(self):
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=0.8))
        for r in (geometry.horizons().r_plus, geometry.horizons().r_minus):
            position = np.array([math.hypot(r, 0.8), 0.0, 0.0])
            self.assertTrue(np.all(np.isfinite(geometry.inverse_metric_derivatives(position))))
        with self.assertRaises(CoordinateDomainError):
            geometry.inverse_metric_derivatives(np.array([0.4, 0.0, 0.0]))

    def test_schwarzschild_inverse_metric_derivative_components(self):
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=0.0))
        radius = 4.0
        derivatives = geometry.inverse_metric_derivatives(np.array([radius, 0.0, 0.0]))
        # At x=r>0, g^tt=-1-2/r; g^tx=2/r.
        self.assertAlmostEqual(derivatives[1, 0, 0], 2.0 / radius**2, places=14)
        self.assertAlmostEqual(derivatives[1, 0, 1], -2.0 / radius**2, places=14)
        self.assertAlmostEqual(derivatives[2, 0, 0], 0.0, places=14)
        self.assertAlmostEqual(derivatives[3, 0, 0], 0.0, places=14)


class HamiltonianTests(unittest.TestCase):
    def test_null_hamiltonian_position_rhs_and_stationarity(self):
        for spin in SPINS:
            geometry = KerrSchildGeometry(SpacetimeParameters(a_star=spin))
            for position in POSITIONS:
                for momentum in SPATIAL_MOMENTA:
                    with self.subTest(spin=spin, position=position, momentum=momentum):
                        state = controlled_null_state(geometry, position, momentum)
                        self.assertLess(abs(hamiltonian_value(geometry, state)), 2e-15)
                        rhs = null_geodesic_rhs(geometry, state)
                        self.assertEqual(rhs.shape, (8,))
                        np.testing.assert_allclose(
                            rhs[:4], geometry.inverse_metric(position) @ state[4:],
                            rtol=0, atol=2e-15,
                        )
                        self.assertEqual(rhs[4], 0.0)  # dp_t/dlambda

    def test_momentum_rhs_against_independent_hamiltonian_gradient(self):
        for spin in SPINS:
            geometry = KerrSchildGeometry(SpacetimeParameters(a_star=spin))
            for position in POSITIONS:
                for momentum in SPATIAL_MOMENTA:
                    with self.subTest(spin=spin, position=position, momentum=momentum):
                        state = controlled_null_state(geometry, position, momentum)
                        analytic = null_geodesic_rhs(geometry, state)[4:]
                        reference = -finite_difference_hamiltonian_gradient(geometry, state)
                        self.assertLess(np.max(np.abs(analytic - reference)), 2e-8)

    def test_schwarzschild_spherical_rotation_covariance(self):
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=0.0))
        position = np.array([4.0, 1.0, 2.0])
        momentum = np.array([0.3, -0.1, 0.7])
        state = controlled_null_state(geometry, position, momentum)
        # A rotation mixing the spin-axis direction z with x checks spherical,
        # rather than only axial, symmetry in the a=0 limit.
        rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        rotated = state.copy()
        rotated[1:4] = rotation @ position
        rotated[5:8] = rotation @ momentum
        original_rhs = null_geodesic_rhs(geometry, state)
        rotated_rhs = null_geodesic_rhs(geometry, rotated)
        self.assertLess(abs(hamiltonian_value(geometry, rotated)), 2e-15)
        np.testing.assert_allclose(rotated_rhs[[0, 4]], original_rhs[[0, 4]], rtol=0, atol=2e-15)
        np.testing.assert_allclose(rotated_rhs[1:4], rotation @ original_rhs[1:4], rtol=0, atol=2e-15)
        np.testing.assert_allclose(rotated_rhs[5:8], rotation @ original_rhs[5:8], rtol=0, atol=2e-15)

    def test_geometry_interface_is_swappable_without_kerr_logic_in_rhs(self):
        class GeometryWrapper(Geometry):
            """Test-only delegation: proves the RHS uses only Geometry API."""

            def __init__(self, base):
                self.base = base

            @property
            def parameters(self):
                return self.base.parameters

            def radial_coordinate(self, position):
                return self.base.radial_coordinate(position)

            def metric(self, position):
                return self.base.metric(position)

            def inverse_metric(self, position):
                return self.base.inverse_metric(position)

            def inverse_metric_derivatives(self, position):
                return self.base.inverse_metric_derivatives(position)

            def horizons(self):
                return self.base.horizons()

            def singularity(self):
                return self.base.singularity()

            def domain(self, position):
                return self.base.domain(position)

        kerr = KerrSchildGeometry(SpacetimeParameters(a_star=0.7))
        wrapper = GeometryWrapper(kerr)
        state = controlled_null_state(kerr, POSITIONS[0], SPATIAL_MOMENTA[0])
        np.testing.assert_array_equal(null_geodesic_rhs(wrapper, state), null_geodesic_rhs(kerr, state))
        self.assertEqual(hamiltonian_value(wrapper, state), hamiltonian_value(kerr, state))

    def test_bad_state_and_unsupported_coordinate_domain_fail_explicitly(self):
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=0.8))
        with self.assertRaisesRegex(ValueError, "eight finite"):
            null_geodesic_rhs(geometry, np.zeros(7))
        with self.assertRaisesRegex(ValueError, "eight finite"):
            hamiltonian_value(geometry, np.full(8, np.nan))
        with self.assertRaises(CoordinateDomainError):
            null_geodesic_rhs(geometry, np.array([0., 0.4, 0., 0., -1., 0., 0., 1.]))


if __name__ == "__main__":
    unittest.main()
