"""Optional fused CPU arithmetic for the existing Kerr-Schild Hamiltonian.

No differential equation or geometry model differs from the reference path.
Numba compiles these scalar formulas for an ODE callback; the public metric,
derivative and pointwise Hamiltonian APIs retain independent Python code.
The native function does not cache any position- or spin-dependent values.
"""

from math import hypot, isfinite, sqrt
from time import perf_counter

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


HAS_NUMBA = njit is not None


def _fused_formula(x, y, z, pt, px, py, pz, a, branch, chart_sign, hemisphere):
    """Analytic RHS; scalar intermediates shared only within this evaluation.

    ``hemisphere=0`` represents the default branch without a disk selector.
    All other arguments are plain numbers, so the same compiled signature
    handles positive/negative sheets and ingoing/outgoing KS patches.
    """
    if not (isfinite(x) and isfinite(y) and isfinite(z) and isfinite(pt)
            and isfinite(px) and isfinite(py) and isfinite(pz)):
        raise ValueError("state must have finite components")
    a2 = a*a
    xy2 = x*x+y*y
    b = xy2+z*z-a2
    root = hypot(b, 2.*a*z)
    if b >= 0.:
        r2 = .5*(b+root)
    else:
        r2 = 2.*a2*z*z/(root-b)
    magnitude = sqrt(r2)
    if hemisphere != 0 and xy2 < a2:
        if z == 0.:
            r = 0.
        else:
            r = hemisphere * (1. if z > 0. else -1.) * magnitude
    else:
        r = branch*magnitude
    if r == 0.:
        if hemisphere == 0 or a == 0. or xy2 >= a2:
            raise ValueError("singularity or regular disk without a hemisphere")
        cosine = hemisphere*sqrt(max(0., 1.-xy2/a2))
    elif a != 0. and xy2 < .9*a2:
        cosine_magnitude = sqrt(max(0., 1.-xy2/(r*r+a2)))
        cosine = cosine_magnitude if z/r > 0. else -cosine_magnitude
    else:
        cosine = z/r

    sigma = r*r+a2*cosine*cosine
    if not isfinite(sigma) or sigma <= 0.:
        raise ValueError("invalid Sigma")
    den = r*r+a2
    inv_den = 1./den
    inv_sigma = 1./sigma
    H = r*inv_sigma
    l1 = (chart_sign*r*x+a*y)*inv_den
    l2 = (chart_sign*r*y-a*x)*inv_den
    l3 = chart_sign*cosine
    L = -pt+l1*px+l2*py+l3*pz
    factor = 2.*H*L
    output = np.empty(8, dtype=np.float64)
    output[0] = -pt+factor
    output[1] = px-factor*l1
    output[2] = py-factor*l2
    output[3] = pz-factor*l3
    output[4] = 0.

    dr0 = r*x*inv_sigma
    dr1 = r*y*inv_sigma
    dr2 = den*cosine*inv_sigma
    if r != 0. and (abs(cosine) < .1 or a == 0.):
        dc0 = -cosine*dr0/r
        dc1 = -cosine*dr1/r
        dc2 = (1.-cosine*dr2)/r
    else:
        dc0 = (-x*inv_den+xy2*r*dr0*inv_den*inv_den)/cosine
        dc1 = (-y*inv_den+xy2*r*dr1*inv_den*inv_den)/cosine
        dc2 = (xy2*r*dr2*inv_den*inv_den)/cosine
    for axis in range(3):
        dr = dr0 if axis == 0 else dr1 if axis == 1 else dr2
        dc = dc0 if axis == 0 else dc1 if axis == 1 else dc2
        dsigma = 2.*r*dr+2.*a2*cosine*dc
        dH = dr*inv_sigma-H*dsigma*inv_sigma
        dden = 2.*r*dr
        dl1 = (chart_sign*dr*x+chart_sign*r*(axis == 0)+a*(axis == 1)
               -l1*dden)*inv_den
        dl2 = (chart_sign*dr*y+chart_sign*r*(axis == 1)-a*(axis == 0)
               -l2*dden)*inv_den
        dl3 = chart_sign*dc
        dL = dl1*px+dl2*py+dl3*pz
        output[5+axis] = dH*L*L+2.*H*dL*L
    return output


if HAS_NUMBA:
    fused_rhs = njit(cache=True, nogil=True)(_fused_formula)
else:
    fused_rhs = None


def warm_compiled_rhs() -> float | None:
    """Compile/load all scalar-argument signatures before the first slider solve.

    Return elapsed seconds, or ``None`` if Numba is unavailable. The caller
    chooses *when* to pay this cost, typically at headless backend startup.
    """
    if fused_rhs is None:
        return None
    start = perf_counter()
    fused_rhs(4., .5, .6, -1., -.2, .1, .3, .7, 1, 1, 0)
    return perf_counter()-start
