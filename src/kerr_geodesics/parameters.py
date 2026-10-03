"""Geometry parameters. G = c = M = 1 throughout this project."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class SpacetimeParameters:
    """Signed a_star and a separately named black-hole charge.

    Q_bh is accepted as data for a future geometry implementation; the
    KerrSchildGeometry constructor explicitly rejects nonzero Q_bh.
    The Carter integral belongs to a *ray* and is named
    carter_constant in ConservedQuantities. It is not a spacetime parameter.
    """

    a_star: float
    Q_bh: float = 0.0
    G: float = 1.0
    c: float = 1.0
    M: float = 1.0

    def __post_init__(self) -> None:
        for name in ("a_star", "Q_bh", "G", "c", "M"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        for name in ("G", "c", "M"):
            if getattr(self, name) != 1.0:
                raise ValueError(f"v0.1A fixes {name}=1")
