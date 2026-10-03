"""Geometry boundary: no integrator, camera, renderer, or solver dependency."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .parameters import SpacetimeParameters


class DomainKind(str, Enum):
    REGULAR = "REGULAR"
    SINGULARITY = "SINGULARITY"
    CHART_BOUNDARY = "CHART_BOUNDARY"


class HorizonRegime(str, Enum):
    SUB_EXTREMAL = "SUB_EXTREMAL"
    EXTREMAL = "EXTREMAL"
    SUPER_EXTREMAL = "SUPER_EXTREMAL"


@dataclass(frozen=True, slots=True)
class DomainInfo:
    kind: DomainKind
    r: float
    sigma: float | None
    explanation: str


@dataclass(frozen=True, slots=True)
class SingularityInfo:
    kind: str  # POINT for a=0, RING otherwise
    ring_radius: float
    plane_z: float = 0.0


@dataclass(frozen=True, slots=True)
class HorizonInfo:
    regime: HorizonRegime
    r_plus: float | None
    r_minus: float | None
    has_distinct_inner_horizon: bool

    @property
    def exists(self) -> bool:
        return self.r_plus is not None


@dataclass(frozen=True, slots=True)
class ConservedQuantities:
    """Killing energy, axial momentum and the null Carter constant.

    Here carter_constant denotes the standard Q = K - (L_z-aE)^2,
    never the black-hole electric charge Q_bh.
    """

    energy: float
    angular_momentum_z: float
    carter_constant: float


class CoordinateDomainError(ValueError):
    def __init__(self, info: DomainInfo):
        self.info = info
        super().__init__(f"{info.kind.value}: {info.explanation}")


class Geometry(ABC):
    """Stationary 3D Cartesian geometry, with spacetime ordering (t,x,y,z).

    All metric and coordinate-domain logic belongs in geometry implementations;
    geodesic equations only consume this interface.
    """

    @property
    @abstractmethod
    def parameters(self) -> SpacetimeParameters:
        pass

    @abstractmethod
    def radial_coordinate(self, position: NDArray[np.float64]) -> float:
        pass

    @abstractmethod
    def metric(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return covariant g_(mu nu), ordering (t, x, y, z)."""

    @abstractmethod
    def inverse_metric(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return contravariant g^(mu nu), ordering (t, x, y, z)."""

    @abstractmethod
    def inverse_metric_derivatives(
        self, position: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return analytic partial_sigma g^(mu nu), shape (4, 4, 4).

        Axis 0 is the derivative coordinate (t,x,y,z). For this stationary
        geometry contract, result[0] is identically zero. No finite
        differences belong in a production geometry implementation.
        """

    @abstractmethod
    def horizons(self) -> HorizonInfo:
        pass

    @abstractmethod
    def singularity(self) -> SingularityInfo:
        pass

    @abstractmethod
    def domain(self, position: NDArray[np.float64]) -> DomainInfo:
        pass

    def conserved_quantities(
        self, position: NDArray[np.float64], momentum: NDArray[np.float64]
    ) -> ConservedQuantities:
        """Null-geodesic invariants when implemented by this geometry.

        Geometry-specific Killing/Carter formulas must not enter initial
        condition or integrator code. A concrete geometry opts in by
        overriding this method. The default preserves v0.1B subclasses.
        """
        raise NotImplementedError("This geometry does not provide null conserved quantities")

    def radial_coordinate_gradient(
        self, position: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Analytic spatial gradient of the geometry's radial coordinate.

        Used to confirm an outward finite-boundary crossing. A concrete
        geometry supporting the integrator must implement this method.
        """
        raise NotImplementedError("This geometry does not provide radial gradient")

    def chart_boundary_coordinate(self, position: NDArray[np.float64]) -> float | None:
        """Positive coordinate tending to zero at unsupported regular chart crossing.

        Return None if the geometry has no such boundary in this chart.
        This is separate from the physical singularity measure DomainInfo.sigma.
        """
        return None

    def disk_extension(self, position: NDArray[np.float64]):
        """Return a geometry with a selected smooth disk continuation, if needed."""
        return None

    def fixed_branch(self, position: NDArray[np.float64]):
        """Return a fixed-sheet geometry after a regular disk crossing."""
        return None

    def horizon_chart_handoff(self, state: NDArray[np.float64], target_chart: str):
        """Return (new geometry, transformed state) between regular KS patches."""
        return None

    def chart_transition_surface(self, state: NDArray[np.float64]):
        """Optional (target_chart, signed-r switch radius) before an incompatible horizon."""
        return None

    def fast_hamiltonian_rhs(self, state: NDArray[np.float64]):
        """Optional compiled evaluation of the *same* full Hamiltonian RHS.

        A geometry may opt in without introducing its formulas into the
        integrator. Returning None keeps the reference implementation.
        """
        return None
