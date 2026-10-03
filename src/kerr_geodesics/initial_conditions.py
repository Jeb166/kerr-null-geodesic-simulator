"""Geometry-agnostic Eulerian frame and local direction -> null state.

The free-camera's future orientation is outside this module. A local
direction is defined relative to the symmetric inverse-square-root triad
of the selected geometry's spatial metric on a Kerr-Schild time slice.
"""

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
from numpy.typing import NDArray

from .geometry import ConservedQuantities, Geometry


FRAME_CONVENTION = "ks_slice_eulerian_symmetric_inverse_sqrt"


@dataclass(frozen=True, slots=True)
class RayDefinition:
    """Reusable local launch data, independent of the chosen spin geometry."""

    coordinate_position: NDArray[np.float64]
    local_direction: NDArray[np.float64]
    coordinate_time: float = 0.0
    omega_local: float = 1.0
    frame_convention: str = FRAME_CONVENTION

    def __post_init__(self):
        pos = np.array(self.coordinate_position, dtype=np.float64, copy=True)
        direction = np.array(self.local_direction, dtype=np.float64, copy=True)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            raise ValueError("coordinate_position must be three finite numbers")
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("local_direction must be three finite numbers")
        if abs(float(np.linalg.norm(direction)) - 1.0) > 1e-12:
            raise ValueError("local_direction must have unit Euclidean norm")
        if not isfinite(self.coordinate_time):
            raise ValueError("coordinate_time must be finite")
        if self.omega_local != 1.0:
            raise ValueError("v0.1C fixes omega_local=1")
        if self.frame_convention != FRAME_CONVENTION:
            raise ValueError(f"unsupported frame_convention: {self.frame_convention}")
        pos.setflags(write=False)
        direction.setflags(write=False)
        object.__setattr__(self, "coordinate_position", pos)
        object.__setattr__(self, "local_direction", direction)


@dataclass(frozen=True, slots=True)
class EulerianFrame:
    observer: NDArray[np.float64]  # n^mu
    triad: NDArray[np.float64]     # e_(hat i)^mu, shape (3,4)
    lapse: float


@dataclass(frozen=True, slots=True)
class InitialConditions:
    definition: RayDefinition
    frame: EulerianFrame
    tangent: NDArray[np.float64]  # k^mu
    momentum: NDArray[np.float64]  # p_mu
    state: NDArray[np.float64]  # (t,x,y,z,p_t,p_x,p_y,p_z)
    local_energy: float
    conserved: ConservedQuantities


def eulerian_frame(geometry: Geometry, position: NDArray[np.float64]) -> EulerianFrame:
    """Unit future normal and Cartesian-aligned orthonormal spatial triad.

    3+1 identities: n_mu=(-alpha,0,0,0),
    alpha=1/sqrt(-g^tt), n^mu=g^(mu nu)n_nu and gamma_ij=g_ij.
    The unique positive-definite symmetric gamma^(-1/2) defines triad
    columns, without sign/eigenvector flips at eigenvalue degeneracies.
    These formulas use Geometry, not Kerr-Schild H or ell.
    """
    inverse = geometry.inverse_metric(position)
    metric = geometry.metric(position)
    if not np.all(np.isfinite(inverse)) or not np.all(np.isfinite(metric)):
        raise ValueError("geometry metric must be finite at observer position")
    if inverse[0, 0] >= 0.0:
        raise ValueError("constant-time slice has no timelike Eulerian normal here")
    lapse = 1.0 / sqrt(-float(inverse[0, 0]))
    normal_covariant = np.array([-lapse, 0.0, 0.0, 0.0])
    observer = inverse @ normal_covariant

    spatial_metric = metric[1:4, 1:4]
    eigenvalues, eigenvectors = np.linalg.eigh(spatial_metric)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("constant-time spatial metric must be positive definite")
    symmetric_inverse_root = (eigenvectors * eigenvalues**(-0.5)) @ eigenvectors.T
    triad = np.zeros((3, 4), dtype=np.float64)
    triad[:, 1:] = symmetric_inverse_root.T
    return EulerianFrame(observer=observer, triad=triad, lapse=lapse)


def create_initial_conditions(geometry: Geometry, definition: RayDefinition) -> InitialConditions:
    """Rebuild metric-dependent data from one saved local launch definition."""
    frame = eulerian_frame(geometry, definition.coordinate_position)
    tangent = definition.omega_local * (
        frame.observer + definition.local_direction @ frame.triad
    )
    momentum = geometry.metric(definition.coordinate_position) @ tangent
    conserved = geometry.conserved_quantities(definition.coordinate_position, momentum)
    state = np.r_[definition.coordinate_time, definition.coordinate_position, momentum]
    local_energy = -float(momentum @ frame.observer)
    return InitialConditions(
        definition=definition, frame=frame, tangent=tangent,
        momentum=momentum, state=state, local_energy=local_energy,
        conserved=conserved,
    )
