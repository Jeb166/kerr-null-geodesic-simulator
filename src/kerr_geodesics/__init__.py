"""Headless Kerr geometry, Hamiltonian RHS, and local null launches."""

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
from .kerr_schild import KerrSchildGeometry
from .hamiltonian import hamiltonian_value, null_geodesic_rhs
from .initial_conditions import (
    FRAME_CONVENTION,
    EulerianFrame,
    InitialConditions,
    RayDefinition,
    create_initial_conditions,
    eulerian_frame,
)
from .integration import (
    EventKind,
    HorizonCrossing,
    IntegrationSettings,
    RegionTransition,
    TerminationStatus,
    TrajectoryResult,
    integrate_ray,
)
from .parameters import SpacetimeParameters
from .parallel import RayBatchExecutor
from .scheduler import (
    ComputeRequest, ComputeSnapshot, ComputeStatus, LatestWinsScheduler,
    RayEntry, RayOutput,
)
from .solving import (
    ReliabilityAssessment,
    ReliabilityLevel,
    assess_trajectory,
    solve_ray_for_spin,
    solve_rays,
    sweep_ray_over_spins,
    warm_compute_backend,
)

__all__ = [
    "ConservedQuantities",
    "CoordinateDomainError",
    "DomainInfo",
    "DomainKind",
    "Geometry",
    "FRAME_CONVENTION",
    "EulerianFrame",
    "EventKind",
    "HorizonInfo",
    "HorizonRegime",
    "HorizonCrossing",
    "IntegrationSettings",
    "RegionTransition",
    "ReliabilityAssessment",
    "ReliabilityLevel",
    "InitialConditions",
    "hamiltonian_value",
    "KerrSchildGeometry",
    "null_geodesic_rhs",
    "RayDefinition",
    "RayBatchExecutor",
    "RayEntry",
    "RayOutput",
    "ComputeRequest",
    "ComputeSnapshot",
    "ComputeStatus",
    "LatestWinsScheduler",
    "TerminationStatus",
    "TrajectoryResult",
    "create_initial_conditions",
    "eulerian_frame",
    "integrate_ray",
    "assess_trajectory",
    "solve_ray_for_spin",
    "solve_rays",
    "sweep_ray_over_spins",
    "warm_compute_backend",
    "SingularityInfo",
    "SpacetimeParameters",
]
