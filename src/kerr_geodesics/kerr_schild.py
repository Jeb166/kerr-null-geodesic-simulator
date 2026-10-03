"""Signed-r ingoing/outgoing Cartesian Kerr-Schild geometry, spin along +z.

Conventions: (-,+,+,+), (t,x,y,z), G=c=M=1.
Primary formula source: SpECTRE KerrSchild class documentation,
https://spectre-code.org/classgr_1_1Solutions_1_1KerrSchild.html
in particular "Spin in the z direction" and the Kerr-Schild identities.
Horizon and ring-singularity conditions: Visser,
https://arxiv.org/pdf/0706.0622
"""

from math import atan2, hypot, sqrt

import numpy as np
from numpy.typing import NDArray

from .geometry import (
    ConservedQuantities,
    CoordinateDomainError,
    DomainInfo,
    DomainKind,
    Geometry,
    HorizonInfo,
    HorizonRegime,
    SingularityInfo,
)
from .parameters import SpacetimeParameters
from ._native import HAS_NUMBA, fused_rhs


_ETA = np.diag(np.array([-1.0, 1.0, 1.0, 1.0], dtype=np.float64))


def _position3(position: NDArray[np.float64]) -> NDArray[np.float64]:
    xyz = np.asarray(position, dtype=np.float64)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("position must have three finite Cartesian coordinates")
    return xyz


class KerrSchildGeometry(Geometry):
    """One Kerr sheet and one principal-null Cartesian KS chart.

    ``disk_hemisphere`` selects a continuous cos(theta) through a regular
    disk crossing; only the integrator's short crossing segment needs it.
    Without this extra datum the exact disk is genuinely double covered.
    """

    def __init__(self, parameters: SpacetimeParameters, *, branch: int = 1,
                 chart: str = "ingoing", disk_hemisphere: int | None = None,
                 backend: str = "auto"):
        if not isinstance(parameters, SpacetimeParameters):
            raise TypeError("parameters must be SpacetimeParameters")
        if parameters.Q_bh != 0.0:
            raise NotImplementedError("KerrSchildGeometry implements only Q_bh=0")
        if branch not in (-1, 1) or chart not in ("ingoing", "outgoing"):
            raise ValueError("branch must be +/-1 and chart ingoing or outgoing")
        if disk_hemisphere not in (None, -1, 1):
            raise ValueError("disk_hemisphere must be None or +/-1")
        if backend not in ("auto", "reference", "compiled"):
            raise ValueError("backend must be auto, reference or compiled")
        if backend == "compiled" and not HAS_NUMBA:
            raise RuntimeError("compiled backend requires numba>=0.67")
        self._parameters = parameters
        self.branch = branch
        self.chart = chart
        self.disk_hemisphere = disk_hemisphere
        self.backend = backend

    def fast_hamiltonian_rhs(self, state: NDArray[np.float64]):
        """Fast geometry-owned RHS; the public pointwise API remains reference.

        Subclasses may override metric/derivatives. They take the reference
        path so their behavior cannot silently be bypassed by this shortcut.
        """
        if type(self) is not KerrSchildGeometry or self.backend == "reference" or not HAS_NUMBA:
            return None
        return fused_rhs(
            float(state[1]), float(state[2]), float(state[3]),
            float(state[4]), float(state[5]), float(state[6]), float(state[7]),
            self.parameters.a_star, self.branch,
            1 if self.chart == "ingoing" else -1,
            self.disk_hemisphere if self.disk_hemisphere is not None else 0,
        )

    @property
    def parameters(self) -> SpacetimeParameters:
        return self._parameters

    def radial_coordinate(self, position: NDArray[np.float64]) -> float:
        x, y, z = _position3(position)
        a = self.parameters.a_star  # M=1, hence a=a_star
        b = x * x + y * y + z * z - a * a
        discriminant_root = hypot(b, 2.0 * a * z)

        # u=r^2 solves u^2-b*u-a^2*z^2=0. For b<0, rationalizing
        # avoids cancellation of two nearly equal positive numbers.
        if b >= 0.0:
            u = 0.5 * (b + discriminant_root)
        else:
            u = (2.0 * a * a * z * z) / (discriminant_root - b)
        magnitude = sqrt(u)
        if self.disk_hemisphere is not None and x*x+y*y < a*a:
            if z == 0.0:
                return 0.0
            return self.disk_hemisphere * (1 if z > 0.0 else -1) * magnitude
        return self.branch * magnitude

    def _cosine(self, position: NDArray[np.float64], r: float) -> float:
        x, y, z = position
        a = self.parameters.a_star
        if r == 0.0:
            if self.disk_hemisphere is None:
                raise ValueError("a hemisphere is needed at the regular disk")
            return self.disk_hemisphere * sqrt(max(0.0, 1.0-(x*x+y*y)/(a*a)))
        if a != 0.0 and x*x+y*y < .9*a*a:
            # The implicit equation is better conditioned than z/r at the disk.
            magnitude = sqrt(max(0.0, 1.0-(x*x+y*y)/(r*r+a*a)))
            return magnitude if z/r > 0 else -magnitude
        return z/r

    def singularity(self) -> SingularityInfo:
        radius = abs(self.parameters.a_star)
        return SingularityInfo(
            kind="POINT" if radius == 0.0 else "RING", ring_radius=radius
        )

    def domain(self, position: NDArray[np.float64]) -> DomainInfo:
        x, y, z = _position3(position)
        r = self.radial_coordinate(np.array([x, y, z], dtype=np.float64))
        a = self.parameters.a_star
        if r == 0.0:
            if a == 0.0 or (z == 0.0 and x * x + y * y == a * a):
                return DomainInfo(
                    DomainKind.SINGULARITY, 0.0, 0.0, "exact singular set"
                )
            if self.disk_hemisphere is None:
                return DomainInfo(DomainKind.CHART_BOUNDARY, 0.0, None,
                                  "exact regular disk needs a hemisphere/branch")
        cosine = self._cosine(np.array([x, y, z]), r)
        sigma = r*r + (a*cosine)**2
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("invalid Sigma at this coordinate position")
        return DomainInfo(DomainKind.REGULAR, r, sigma,
                          f"{self.chart} chart, {'positive' if r >= 0 else 'negative'}-r")

    def horizons(self) -> HorizonInfo:
        a_abs = abs(self.parameters.a_star)
        if a_abs > 1.0:
            return HorizonInfo(HorizonRegime.SUPER_EXTREMAL, None, None, False)
        if a_abs == 1.0:
            return HorizonInfo(HorizonRegime.EXTREMAL, 1.0, 1.0, False)

        # (1-|a|)(1+|a|) avoids forming 1-a^2 near extremality;
        # r_minus=a^2/r_plus avoids cancellation near a=0.
        root = sqrt((1.0 - a_abs) * (1.0 + a_abs))
        r_plus = 1.0 + root
        r_minus = (a_abs * a_abs) / r_plus
        return HorizonInfo(
            HorizonRegime.SUB_EXTREMAL,
            r_plus,
            r_minus,
            0.0 < a_abs < 1.0,
        )

    def kerr_schild_fields(
        self, position: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return H and covector ell_mu for g=eta+2H ell ell.

        Sigma=r^2+a^2*cos(theta)^2, H=r/Sigma. For the outgoing
        patch the radial parts of ell reverse sign; the twist keeps a.
        The Minkowski-raised ell is also null for the full metric.
        """
        x, y, z = _position3(position)
        info = self.domain(np.array([x, y, z], dtype=np.float64))
        if info.kind != DomainKind.REGULAR:
            raise CoordinateDomainError(info)
        r = info.r
        a = self.parameters.a_star
        sign = 1 if self.chart == "ingoing" else -1
        denominator = r * r + a * a
        cosine = self._cosine(np.array([x, y, z]), r)
        ell = np.array(
            [
                1.0,
                (sign*r*x+a*y)/denominator,
                (sign*r*y-a*x)/denominator,
                sign*cosine,
            ],
            dtype=np.float64,
        )
        return r / info.sigma, ell

    def metric(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        H, ell_covariant = self.kerr_schild_fields(position)
        return _ETA + 2.0 * H * np.outer(ell_covariant, ell_covariant)

    def inverse_metric(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        H, ell_covariant = self.kerr_schild_fields(position)
        ell_contravariant = _ETA @ ell_covariant
        return _ETA - 2.0 * H * np.outer(ell_contravariant, ell_contravariant)

    def inverse_metric_derivatives(
        self, position: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Analytic derivatives of the inverse metric in (t,x,y,z) order.

        The implicit radial equation is
            r^4 - (x^2+y^2+z^2-a^2) r^2 - a^2 z^2 = 0.
        Its spatial gradient is
            d_i r = (r*x_i + delta_iz*a^2*z/r) / Sigma,
            Sigma = r^2+a^2*z^2/r^2.
        Differentiate H=r/Sigma and the Kerr-Schild ell components in
        kerr_schild_fields with the product/quotient rules, then use
            d_i g^(mu nu) = -2 [(d_i H) ell^mu ell^nu
              + H (d_i ell^mu ell^nu + ell^mu d_i ell^nu)].
        Minkowski raising is constant, so d_i ell^mu = eta^(mu nu) d_i ell_nu.
        This shares the v0.1A metric definitions; it never differentiates
        an inverse-matrix algorithm or evaluates a finite difference.
        """
        x, y, z = _position3(position)
        info = self.domain(np.array([x, y, z], dtype=np.float64))
        if info.kind != DomainKind.REGULAR:
            raise CoordinateDomainError(info)

        r, sigma = info.r, info.sigma
        a = self.parameters.a_star
        a2 = a * a
        r2 = r * r
        denominator = r2 + a2
        H, ell = self.kerr_schild_fields(np.array([x, y, z], dtype=np.float64))

        # Implicit differentiation avoids differentiating the two branches
        # of the numerically stabilized quadratic formula for r^2.
        cosine = self._cosine(np.array([x, y, z]), r)
        sign = 1 if self.chart == "ingoing" else -1
        dr = np.array([r*x, r*y, denominator*cosine]) / sigma
        if r != 0.0 and (abs(cosine) < .1 or a == 0.0):
            dc = (np.array([0., 0., 1.])-cosine*dr)/r
        else:
            # Differentiate c^2=1-(x^2+y^2)/(r^2+a^2). Finite at disk.
            xy2 = x*x+y*y
            dc = (-np.array([x, y, 0.])/denominator
                  +xy2*r*dr/(denominator*denominator))/cosine
        dsigma = 2*r*dr+2*a2*cosine*dc
        dH = dr / sigma - H * dsigma / sigma

        dell = np.zeros((3, 4), dtype=np.float64)
        for axis in range(3):
            ddenominator = 2.0 * r * dr[axis]
            dell[axis, 1] = (
                sign*dr[axis]*x + sign*r*(axis == 0) + a*(axis == 1)
                - ell[1] * ddenominator
            ) / denominator
            dell[axis, 2] = (
                sign*dr[axis]*y + sign*r*(axis == 1) - a*(axis == 0)
                - ell[2] * ddenominator
            ) / denominator
            dell[axis, 3] = sign*dc[axis]

        raised_ell = _ETA @ ell
        raised_dell = dell @ _ETA
        result = np.zeros((4, 4, 4), dtype=np.float64)
        for axis in range(3):
            result[axis + 1] = -2.0 * (
                dH[axis] * np.outer(raised_ell, raised_ell)
                + H * (
                    np.outer(raised_dell[axis], raised_ell)
                    + np.outer(raised_ell, raised_dell[axis])
                )
            )
        # The metric is stationary: d_t g^(mu nu) is exactly zero.
        return result

    def conserved_quantities(
        self, position: NDArray[np.float64], momentum: NDArray[np.float64]
    ) -> ConservedQuantities:
        """E=-p_t, L_z=x*p_y-y*p_x, standard null Carter Q.

        Spheroidal coordinates obey x^2+y^2=(r^2+a^2)sin^2(theta) and
        z=r*cos(theta). The separated null Carter constant is
          Q=p_theta^2 + cos^2(theta)*(L_z^2/sin^2(theta)-a^2 E^2).
        Using p_theta=cos(theta)/sin(theta)*(x*p_x+y*p_y)
                      -r*sin(theta)*p_z,
        and (x*p_x+y*p_y)^2+L_z^2=(x^2+y^2)*(p_x^2+p_y^2),
        expand to remove the apparent polar 0/0. This remains finite on
        the spin axis; it is algebraically the same invariant away from it.
        See Bakun et al. (2024), arxiv:2409.03722, Eq. 9b;
        our Q is their K-(L_z-aE)^2, specialized to null momenta.
        """
        x, y, z = _position3(position)
        p = np.asarray(momentum, dtype=np.float64)
        if p.shape != (4,) or not np.all(np.isfinite(p)):
            raise ValueError("momentum must have four finite covariant components")
        info = self.domain(np.array([x, y, z], dtype=np.float64))
        if info.kind != DomainKind.REGULAR:
            raise CoordinateDomainError(info)
        r, a = info.r, self.parameters.a_star
        cosine = self._cosine(np.array([x, y, z]), r)
        sine_squared = (x * x + y * y) / (r * r + a * a)
        energy = -p[0]
        angular_momentum_z = x * p[2] - y * p[1]
        xy_projection = x * p[1] + y * p[2]
        carter_constant = (
            cosine**2 * ((r * r + a * a) * (p[1]**2 + p[2]**2) - a * a * energy**2)
            - 2.0 * r * cosine * xy_projection * p[3]
            + r * r * sine_squared * p[3]**2
        )
        return ConservedQuantities(
            energy=float(energy), angular_momentum_z=float(angular_momentum_z),
            carter_constant=float(carter_constant),
        )

    def radial_coordinate_gradient(
        self, position: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        x, y, z = _position3(position)
        info = self.domain(np.array([x, y, z]))
        if info.kind != DomainKind.REGULAR:
            raise CoordinateDomainError(info)
        r, a = info.r, self.parameters.a_star
        cosine = self._cosine(np.array([x, y, z]), r)
        return np.array([r*x, r*y, (r*r+a*a)*cosine], dtype=np.float64) / info.sigma

    def chart_boundary_coordinate(self, position: NDArray[np.float64]) -> float | None:
        return None

    def disk_extension(self, position: NDArray[np.float64]):
        """Select a smooth short segment before a regular disk crossing."""
        if self.disk_hemisphere is not None or self.parameters.a_star == 0.0:
            return None
        x, y, z = _position3(position)
        a = abs(self.parameters.a_star)
        r = self.radial_coordinate(position)
        if (z != 0.0 and x*x+y*y < (1.-1e-7)*a*a
                and 0 < abs(r) < .2*a):
            hemisphere = 1 if z/r > 0 else -1
            return KerrSchildGeometry(self.parameters, branch=self.branch,
                                      chart=self.chart, disk_hemisphere=hemisphere,
                                      backend=self.backend)
        return None

    def fixed_branch(self, position: NDArray[np.float64]):
        """Finish the short crossing segment on its new sheet."""
        if self.disk_hemisphere is None:
            return None
        r = self.radial_coordinate(position)
        if r != 0.0 and (1 if r > 0 else -1) != self.branch:
            return KerrSchildGeometry(self.parameters, branch=-self.branch,
                                      chart=self.chart, backend=self.backend)
        return None

    def chart_transition_surface(self, state: NDArray[np.float64]):
        """Choose outgoing KS in block II if the inward Cauchy branch needs it.

        At a horizon the radial potential satisfies R=P^2, where
        P=E(r^2+a^2)-a Lz. The ingoing KS patch cancels the divergent
        coordinate numerator for inward motion only if P(r_h)>0.
        P(r_minus)<0 selects the outgoing patch before inner crossing;
        the halfway radius maximizes |Delta| between both horizons.
        """
        horizon = self.horizons()
        if self.chart != "ingoing" or not horizon.has_distinct_inner_horizon:
            return None
        a = self.parameters.a_star
        p = state[4:8]
        x, y = state[1:3]
        energy = -p[0]
        angular_momentum_z = x*p[2] - y*p[1]
        p_inner = energy*(horizon.r_minus**2+a*a)-a*angular_momentum_z
        if p_inner >= 0.0:
            return None
        return "outgoing", (horizon.r_plus+horizon.r_minus)/2.0

    def horizon_chart_handoff(self, state: NDArray[np.float64], target_chart: str):
        """Canonical transformation between horizon-penetrating KS patches.

        At the switch radius phi_out=phi_in, so the Cartesian coordinates
        rotate by -2 atan2(a,r). Differentiating the complete chart map
        supplies the Jacobian for the tangent; p_out=g_out J g_in^-1 p_in.
        Switches occur on the safe side of the next horizon, at finite Delta.
        """
        if (self.chart == target_chart or target_chart not in ("ingoing", "outgoing")
                or self.parameters.a_star == 0.0):
            return None
        info = self.domain(state[1:4]); horizon = self.horizons()
        if info.r <= 0:
            return None
        r, a = info.r, self.parameters.a_star
        delta = r*r-2*r+a*a
        if abs(delta) < 1e-7:
            return None
        if horizon.exists:
            if target_chart == "outgoing":
                if r >= horizon.r_plus:
                    return None
                if r >= horizon.r_minus and self.chart_transition_surface(state) is None:
                    return None
            elif r <= horizon.r_plus:
                return None
        # For super-extremal Kerr, Delta>0 everywhere. The same two regular
        # KS patches are useful on opposite sides of an inner radial turn.
        if not horizon.exists and ((target_chart == "outgoing" and r >= 1.0)
                                   or (target_chart == "ingoing" and r <= 1.0)):
            return None
        direction = 1 if target_chart == "outgoing" else -1
        angle = -direction*2*atan2(a, r)
        co, si = np.cos(angle), np.sin(angle)
        rotation = np.array([[co,-si],[si,co]])
        transformed = np.array(state, copy=True)
        transformed[1:3] = rotation @ state[1:3]
        dr = self.radial_coordinate_gradient(state[1:4])
        jacobian = np.eye(4)
        jacobian[0, 1:4] = -direction*4*r*dr/delta
        jacobian[1:3, 1:3] = rotation
        rotated_xy = rotation @ state[1:3]
        angular_slope = direction*(2*a/(r*r+a*a)-2*a/delta)
        jacobian[1:3, 1:4] += np.outer(
            angular_slope*np.array([-rotated_xy[1], rotated_xy[0]]), dr
        )
        target = KerrSchildGeometry(self.parameters, branch=1,
                                   chart=target_chart, backend=self.backend)
        tangent = self.inverse_metric(state[1:4]) @ state[4:8]
        transformed[4:8] = target.metric(transformed[1:4]) @ jacobian @ tangent
        return target, transformed

    def canonical_ingoing_position(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Spatial point in the initial ingoing Cartesian chart for display.

        A chart handoff rotates the Cartesian x/y coordinates. Undoing that
        rotation here prevents a viewer from joining different charts as if
        they used the same spatial coordinates. Physics states are untouched.
        """
        point = np.array(position, dtype=np.float64, copy=True)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("position must contain three finite coordinates")
        if self.chart == "outgoing":
            angle = 2.0 * atan2(self.parameters.a_star, self.radial_coordinate(point))
            co, si = np.cos(angle), np.sin(angle)
            point[0], point[1] = co*point[0]-si*point[1], si*point[0]+co*point[1]
        point.setflags(write=False)
        return point
