"""Independent geometric checks; no geodesic integration in v0.1A."""

import math
import unittest

import numpy as np

from kerr_geodesics import (
    CoordinateDomainError,
    DomainKind,
    Geometry,
    HorizonRegime,
    KerrSchildGeometry,
    SpacetimeParameters,
)


ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


class ParameterAndBoundaryTests(unittest.TestCase):
    def test_units_and_charge_boundary(self):
        params = SpacetimeParameters(a_star=0.7)
        self.assertEqual((params.G, params.c, params.M, params.Q_bh), (1, 1, 1, 0))
        self.assertIsInstance(KerrSchildGeometry(params), Geometry)
        with self.assertRaisesRegex(ValueError, "M=1"):
            SpacetimeParameters(a_star=0.7, M=2)
        with self.assertRaisesRegex(ValueError, "finite"):
            SpacetimeParameters(a_star=float("nan"))
        future_parameters = SpacetimeParameters(a_star=0.7, Q_bh=0.1)
        with self.assertRaisesRegex(NotImplementedError, "Q_bh=0"):
            KerrSchildGeometry(future_parameters)

    def test_singularity_is_a_ring_or_a_point_not_a_disk(self):
        kerr = KerrSchildGeometry(SpacetimeParameters(a_star=0.8))
        self.assertEqual(kerr.singularity().ring_radius, 0.8)
        self.assertEqual(kerr.domain(np.array([0.8, 0.0, 0.0])).kind, DomainKind.SINGULARITY)
        self.assertEqual(kerr.domain(np.array([0.4, 0.0, 0.0])).kind, DomainKind.CHART_BOUNDARY)
        self.assertEqual(kerr.domain(np.array([0.4, 0.0, 0.1])).kind, DomainKind.REGULAR)
        with self.assertRaises(CoordinateDomainError):
            kerr.metric(np.array([0.4, 0.0, 0.0]))
        schwarzschild = KerrSchildGeometry(SpacetimeParameters(a_star=0))
        self.assertEqual(schwarzschild.singularity().kind, "POINT")
        self.assertEqual(
            schwarzschild.domain(np.zeros(3)).kind, DomainKind.SINGULARITY
        )

    def test_horizon_regimes_and_regular_inner_horizon(self):
        nonrotating = KerrSchildGeometry(SpacetimeParameters(a_star=0)).horizons()
        self.assertEqual((nonrotating.r_plus, nonrotating.r_minus), (2.0, 0.0))
        self.assertFalse(nonrotating.has_distinct_inner_horizon)
        regular = KerrSchildGeometry(SpacetimeParameters(a_star=-0.8)).horizons()
        self.assertEqual(regular.regime, HorizonRegime.SUB_EXTREMAL)
        self.assertAlmostEqual(regular.r_plus, 1.6)
        self.assertAlmostEqual(regular.r_minus, 0.4)
        self.assertTrue(regular.has_distinct_inner_horizon)
        self.assertAlmostEqual(regular.r_plus * regular.r_minus, 0.8**2)
        extremal = KerrSchildGeometry(SpacetimeParameters(a_star=1.0)).horizons()
        self.assertEqual(extremal.regime, HorizonRegime.EXTREMAL)
        self.assertEqual((extremal.r_plus, extremal.r_minus), (1.0, 1.0))
        self.assertFalse(extremal.has_distinct_inner_horizon)
        naked = KerrSchildGeometry(SpacetimeParameters(a_star=1.001)).horizons()
        self.assertEqual(naked.regime, HorizonRegime.SUPER_EXTREMAL)
        self.assertIsNone(naked.r_plus)
        self.assertIsNone(naked.r_minus)


class KerrSchildIdentityTests(unittest.TestCase):
    def test_radial_coordinate_obeys_independent_spheroidal_equation(self):
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=0.7))
        for xyz in (
            np.array([2.0, 1.0, 0.4]),
            np.array([0.2, 0.0, 1e-12]),  # cancellation-prone disk vicinity
            np.array([0.0, 0.0, 3.0]),
        ):
            with self.subTest(xyz=xyz):
                r = geometry.radial_coordinate(xyz)
                self.assertGreater(r, 0.0)
                lhs = (
                    (xyz[0] ** 2 + xyz[1] ** 2) / (r * r + 0.7**2)
                    + (xyz[2] / r) ** 2
                )
                self.assertAlmostEqual(lhs, 1.0, places=12)

    def test_metric_inverse_null_rank_one_and_determinant(self):
        rng = np.random.default_rng(5903)
        samples = [rng.uniform(-5, 5, size=3) for _ in range(35)]
        samples.extend(
            [
                np.array([0.0, 0.0, 1.0]),
                np.array([math.sqrt(1.6**2 + 0.8**2), 0.0, 0.0]),
                np.array([0.2, 0.0, 1e-5]),
            ]
        )
        for spin in (0.0, -0.8, 0.8, 0.9999, 1.0, 1.05):
            geometry = KerrSchildGeometry(SpacetimeParameters(a_star=spin))
            for xyz in samples:
                with self.subTest(spin=spin, xyz=xyz):
                    H, ell = geometry.kerr_schild_fields(xyz)
                    raised = ETA @ ell
                    g = geometry.metric(xyz)
                    g_inv = geometry.inverse_metric(xyz)
                    self.assertAlmostEqual(ell @ raised, 0.0, places=11)
                    self.assertLess(np.max(np.abs(g @ g_inv - np.eye(4))), 2e-12)
                    self.assertLess(np.max(np.abs(g - g.T)), 1e-14)
                    self.assertLess(np.max(np.abs(g_inv - g_inv.T)), 1e-14)
                    self.assertLess(np.max(np.abs(g - ETA - 2*H*np.outer(ell, ell))), 2e-13)
                    self.assertAlmostEqual(np.linalg.det(g), -1.0, places=10)
                    self.assertAlmostEqual(raised @ g @ raised, 0.0, places=10)

    def test_schwarzschild_limit_has_known_cartesian_ks_metric(self):
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=0))
        xyz = np.array([4.0, 0.0, 0.0])
        self.assertAlmostEqual(geometry.radial_coordinate(xyz), 4.0)
        expected = np.array(
            [[-0.5, 0.5, 0, 0],
             [0.5, 1.5, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=float
        )
        np.testing.assert_allclose(geometry.metric(xyz), expected, rtol=0, atol=1e-15)

    def test_metric_is_finite_at_outer_and_inner_horizon(self):
        geometry = KerrSchildGeometry(SpacetimeParameters(a_star=0.8))
        horizon = geometry.horizons()
        for radius in (horizon.r_plus, horizon.r_minus):
            xyz = np.array([math.sqrt(radius * radius + 0.8**2), 0.0, 0.0])
            with self.subTest(radius=radius):
                self.assertAlmostEqual(geometry.radial_coordinate(xyz), radius, places=13)
                g = geometry.metric(xyz)
                g_inv = geometry.inverse_metric(xyz)
                self.assertTrue(np.all(np.isfinite(g)))
                self.assertLess(np.max(np.abs(g @ g_inv - np.eye(4))), 2e-14)


if __name__ == "__main__":
    unittest.main()
