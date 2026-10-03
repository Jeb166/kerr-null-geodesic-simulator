# v0.1A geometry validation

Checkpoint update (user-reported, Windows 11 / Python 3.14.4 / NumPy
2.5.3 / SciPy 1.18.1): v0.2A passed 67/67, and the 49 reference
trajectory comparisons matched. Earlier v0.2A Windows validation debt
recorded later in this historical log is closed. The *new viewer*
Windows installation and interaction check remains open.

Executed on 2026-09-20 with Python 3.12.14 and NumPy 2.3.5.

Command: PYTHONPATH=src python -m unittest discover -s tests -v

Result: Ran 7 tests in 0.064s — OK.

The tests cover fixed units and rejected charge in the Kerr model; separate
ring, disk and Schwarzschild-point domain classifications; horizon regimes;
the defining oblate-spheroidal equation for r; Kerr-Schild nullness;
metric symmetry, inverse and determinant; known Schwarzschild components;
and finite metric values at both sub-extremal horizon radii.

Command: PYTHONPATH=src python scripts/geometry_diagnostics.py

Results with a fixed random seed, 200 positions at each of 5 spins
(a_star=0, 0.8, 0.9999, 1.0, 1.05), 1000 samples total:

    max |g @ g_inverse - I|  = 1.554312e-15
    max |ell @ eta @ ell|    = 6.377592e-16
    max |det(g) + 1|         = 6.661338e-16

This establishes algebraic and selected coordinate-domain checks only.
No initial null 4-momentum, geodesic, integrator, ray horizon crossing,
conserved quantity, or Kerr-Newman physics exists in v0.1A.

# v0.1B analytic derivatives and pointwise Hamiltonian RHS

Executed on 2026-09-20 with Python 3.12.14 and NumPy 2.3.5.

Command: `PYTHONPATH=src python -m unittest discover -s tests -v`

Result: **15 tests, all passing**, including all **7 unchanged v0.1A tests**.

Command: `PYTHONPATH=src python scripts/v0_1b_diagnostics.py`

Fixed sample: `a_star=(0.0, 0.7, -0.8, 0.999)`, 7 positions per spin,
2 future-directed null covectors per position (28 metric-derivative and
56 Hamiltonian RHS cases). Central-difference spatial step:
`h_i = 1e-5 * max(1, abs(x_i))`; the time step has the same rule.

    max |analytic partial_sigma g^(mu nu) - FD| = 3.982471019626e-10
    max |analytic dp_mu/dlambda + FD partial_mu H_geo| = 1.782102243553e-10
    max |dp_t/dlambda| = 0.000000000000e+00
    max |H_geo| for test null states = 3.033598581825e-16

Absolute acceptance limits fixed before measuring: `2e-8` for both
finite-difference comparisons and `2e-15` for null-Hamiltonian values;
`dp_t/dlambda` must equal zero. The test also checks the analytic tensor
shape and symmetry, finite derivatives at both Kerr horizons, explicit
domain errors, known Schwarzschild derivative components, SO(3) rotation
covariance of the Schwarzschild RHS, and independent geometry-interface
delegation. No results outside the sampled coordinate region, integration,
horizon crossing by rays, or Kerr-Newman equations are claimed.

# v0.1C local-frame/null-launch validation

Executed on 2026-09-20 with Python 3.12.14 and NumPy 2.3.5.
Command: `PYTHONPATH=src python scripts/v0_1c_diagnostics.py`

Fixed sample: five spins `(0, 0.7, -0.8, 0.999, 1.05)`, five positions
per spin (including just outside the outer horizon and a positive-r
position within it), and four local directions, 100 launches total.
All **22/22** unit tests passed (the original 15 unchanged).

| Maximum absolute discrepancy | Measured |
| --- | ---: |
| `g(n,n)+1` | 6.661338147751e-16 |
| `g(e_i,e_j)-delta_ij` | 1.332267629550e-15 |
| `g(n,e_i)` | 6.661338147751e-16 |
| `g(k,k)` | 1.619914762805e-15 |
| `g^(-1)(p,p)` | 2.470964230520e-15 |
| `H_geo` | 1.235482115260e-15 |
| `E_local-1` | 1.110223024625e-15 |

Examples at `(x,y,z)=(3,1,0.4)` for the fourth saved direction:

| `a_star` | `E` | `L_z` | `carter_constant` |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.665651483456 | -1.544330195331 | 7.188061142397 |
| 0.7 | 0.606851059950 | -1.611404976371 | 6.832291953466 |
| 0.999 | 0.576481217131 | -1.654806610576 | 6.463666962078 |

The tests also evaluate the separated Carter formula away from the axis,
the finite axis expression, Schwarzschild angular momentum identity,
spin-recomputed initial momenta, and triad continuity at the pole.
These are local conditions; v0.1C performed no propagation.

# v0.1D single-ray validation

Executed on 2026-09-20 with Python 3.12.14, NumPy 2.3.5 and
SciPy 1.17.0. Commands:

    PYTHONPATH=src python -m unittest discover -s tests -v
    PYTHONPATH=src python scripts/geometry_diagnostics.py
    PYTHONPATH=src python scripts/v0_1b_diagnostics.py
    PYTHONPATH=src python scripts/v0_1c_diagnostics.py
    PYTHONPATH=src python scripts/v0_1d_diagnostics.py

**34/34 tests passed**: 7 unchanged v0.1A, 8 unchanged v0.1B,
7 unchanged v0.1C, and 12 v0.1D tests. v0.1A's 1,000-sample maximum
`|g*g^(-1)-I|` remained 1.554312e-15; v0.1B's maximum derivative
and RHS finite-difference errors remained 3.982471019626e-10 and
1.782102243553e-10. v0.1C maxima remained as above.

Default DOP853 settings were `rtol=1e-10`, component `atol` of 1e-11
for coordinate time and 1e-12 for spatial coordinates/momenta,
`max_step=0.5`, `epsilon_sigma=1e-4`, and
`chart_boundary_radius=1e-3`. The scenarios below use `r_escape=8`
except the inclined exterior ray (`r_escape=9`).

| Ray (`a_star`) | Status | End affine `lambda` | Accepted steps | Horizon events | Max scaled null residual | Max `L_z` / `carter_constant` drift |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Schwarzschild radial outward (0) | `ESCAPED` | 9.797958971138 | 22 | none | 2.260126e-13 | 0 / 0 |
| Schwarzschild radial inward (0) | `SINGULARITY_APPROACH` | 3.257821357902 | 37 | outer at 1.632993162 | 1.182816e-16 | 0 / 0 |
| Kerr equatorial inward (0.7) | `SINGULARITY_APPROACH` | 2.479539607825 | 78 | outer at 1.729780650; inner at 2.463122834 | 1.798309e-12 | 3.911815e-12 / 0 |
| Kerr axis inward (0.8) | `CHART_BOUNDARY` | 2.354488688581 | 49 | outer at 1.099127764; inner at 2.041237276 | 2.912785e-16 | 0 / 4.440892e-16 |
| Near-extremal exterior (0.999) | `ESCAPED` | 9.855261284252 | 22 | none | 1.136074e-14 | 1.026401e-13 / 1.171285e-14 |
| Super-extremal exterior (1.05) | `ESCAPED` | 9.868905231226 | 22 | none | 1.024604e-14 | 1.060263e-13 / 1.326717e-14 |
| Inclined Kerr exterior (0.7) | `ESCAPED` | 11.663085717602 | 25 | none | 1.586727e-14 | 3.286260e-14 / 5.284662e-14 |

The Schwarzschild inward ray continued beyond `r_+=2` to `r=0.01`,
where `Sigma=1e-4` by definition; reducing `epsilon_sigma` to
2.5e-5 moves the stop to `r=0.005` without changing the outer-horizon
event. The Kerr axis ray stops at `r=0.001` while `Sigma=0.640001`:
that is a regular disk chart guard, not ring-singularity approach.

A separate `r=3M` Schwarzschild circular null orbit stayed within
1.332268e-15 of the radial reference over affine parameter 2, with
`L_z/E=5.196152422707=3*sqrt(3)` (within shown precision). An inward
ray starting outside `r_escape=5` crossed that radius inward and was
**not** called escaped. Forced derivative failure returned
`NUMERICAL_ERROR` with the underlying error message; step and affine
budgets returned `MAX_INTEGRATION_LIMIT`.

Convergence: at `rtol=1e-9` versus `1e-11` for Schwarzschild outward,
end-`lambda` difference was 2.222222e-12 and maximum terminal-state
component difference 1.646683e-12; for Kerr equatorial inward they were
4.071410e-12 and 1.101453e-07 near the singularity guard. Both
classification labels and horizon-crossing counts agreed, with horizon
root times differing by at most 4.5e-12 in the Kerr case.

**Explicit failing sample and limitation:** an inclined Kerr plunge at
`a_star=0.7`, starting `(4,0.5,0.6)` in local direction `(-1,0,0)`,
appears to approach an unsupported inner-horizon branch. DOP853 returned
`NUMERICAL_ERROR: Required step size is less than spacing between numbers`
at affine parameter 3.385864199108 after 316 accepted steps.
Its raw null residual had grown to 5.071674e15, `L_z` drift to
5.293908e+01 and `carter_constant` drift to 5.368709e+08.
**Those final states are not trustworthy physics.** We neither call this
a ring hit nor a chart crossing with a verified root; branch-aware chart
continuation needs separate work. This diagnostic scenario is intentionally
printed, not hidden among the passing test cases.

SciPy's public DOP853 API reports accepted steps through `step()` and
`nfev`, but does not expose a reliable rejected-step count; it is stored
as `None`. Sign-change event detection can miss multiple roots within
one step, and `minimum_sampled_*` refers only to stored samples/events.
The user independently verified v0.1D on Windows 11 with Python 3.14.4,
NumPy 2.5.3 and SciPy 1.18.1: 34/34 tests passed. The v0.1E checks
below are from the separate environment explicitly listed there.

# v0.1E signed-r extension and inner-region robustness

Executed on 2026-09-20 with Python 3.12.14, NumPy 2.3.5 and SciPy
1.17.0. This environment differs from the user's independently checked
Windows 11 / Python 3.14.4 / NumPy 2.5.3 / SciPy 1.18.1 v0.1D setup;
v0.1E has not yet been run there.

    PYTHONPATH=src python -m unittest discover -s tests -v
    PYTHONPATH=src python scripts/v0_1e_diagnostics.py

**48/48 tests passed:** the original 34 checks by intent, with the 33
unchanged v0.1A–D tests and one necessary v0.1D test revision,
plus 14 new signed-region/packaging tests. The revised old test expected
`CHART_BOUNDARY` for a regular axis disk crossing. That expectation is
physically invalid now that signed-r continuation is implemented; it
instead verifies a reliable `ESCAPED` result after the negative sheet.
The seven original v0.1A geometry tests remain unchanged and passing.

The 54 derivative positions include 28 original positive-sheet points,
24 negative-sheet points (both principal-null patches; spins
`0.7, 0.999, 1.05, 1.2`), and two exact regular-disk hemisphere limits.
Independent central differences use
`h_i=1e-5*max(1,abs(x_i))`, with covariant momentum fixed. The 28
original positive-sheet points exercise two future-null momenta each;
negative-sheet Hamiltonian checks use eight future-null states. The
same **absolute `2e-8` tolerances** apply to derivative and RHS finite
differences; no finite difference is evaluated on the production RHS.

| Maximum measured error | v0.1E result |
| --- | ---: |
| `|analytic partial g^(-1) - FD|` | 7.230163134864e-10 |
| `|analytic dp/dlambda + FD partial H_geo|` | 2.127439557853e-09 |
| `|dp_t/dlambda|` | 0 |
| negative-sheet `|g*g^(-1)-I|` | 6.490482390023e-16 |
| negative-sheet `|det(g)+1|` | 4.440892098501e-16 |

The same analytical expressions extend smoothly to the regular disk:
with `a=0.8`, `(x,y,z)=(0.2,0.1,0)`, `Sigma=0.59`, metric = Minkowski,
while `(0.8,0,0)` is the singular ring. A 4-dimensional independent
finite-difference Jacobian (numerical quadratures for both radial
integrals) additionally validates the chart handoff's transformed
inverse metric, tangent, null Hamiltonian and `E/L_z/Carter` values.

## Root cause of the old inclined-ray failure

For `a=0.7`, `x0=(4,0.5,0.6)`, `d_local=(-1,0,0)`, the v0.1D result
crossed the outer horizon inward at `lambda=1.874219672` and the inner
horizon inward at `3.104989890`. It did **not** approach the disk.
The independent separated radial potential

    R(r)=[E(r²+a²)-a L_z]²-(r²-2r+a²)[Carter+(L_z-aE)²]

has a turning root at `r=0.219026167950`; the integrated radial
minimum was `r=0.219026167947`. The ray then headed **outward** toward
the inner horizon `r_-=0.285857157146`, where `R(r_-)=0.0580316181`
and a finite outward radial derivative is expected. The ingoing chart
does not represent this opposite horizon branch smoothly. At its final
v0.1D step `Sigma=0.132597849`, the inverse-metric condition number was
approximately `76.35`, and the radial quartic residual was
`2.6e-18`, while max covariant momentum grew to `6.397e12`. The
analytic-vs-FD metric derivative error there decreases from
`2.05e-7` for FD step `1e-5` to `2.59e-9` for FD step `1e-6`
(second-order truncation in this strongly varying region). Thus neither
ring/disk contact, radial inversion nor divergent metric conditioning
explains the exploding momenta. Ingoing-chart state components diverge
at the **outward inner-horizon branch**, making affine steps fall below
float spacing; DOP853's error is a consequence of this coordinate
representation. This behavior agrees with the radial-direction
restriction discussed by
[Bakun et al.](https://arxiv.org/html/2409.03722v2).

| Diagnostic | v0.1D ingoing only | v0.1E with chart handoff |
| --- | ---: | ---: |
| Status | `NUMERICAL_ERROR` | `ESCAPED` |
| End `lambda` | 3.385864199108 | 9.558756916298 |
| Regions/events | outer IN; inner IN; failed approaching inner OUT | outer IN; inner IN; inner OUT; outer OUT; outer region |
| Handoff | none | ingoing → outgoing at `lambda=3.287134110` |
| Minimum sampled `|r|` | 0.219126486 (no turn-event sample) | 0.219026168 |
| Minimum sampled `Sigma` | 0.132597849 | 0.131156373 |
| Maximum scaled null residual | 1.351539e-11 (scale masks exploding p) | 6.099288e-11 |
| Maximum raw null residual | 5.071674e15 | 4.327243e-10 |
| Maximum absolute `L_z` drift | 5.293908e+01 | 3.547962e-11 |
| Maximum absolute Carter drift | 5.368709e+08 | 3.944101e-11 |
| Accepted steps / RHS evaluations | 316 / 7226 | 81 / 1591 |
| Minimum accepted step | 4.884981e-15 | 8.663087e-03 |

The former final-state scaled-null statistic was misleading because
the normalization grows quadratically with its divergent momentum;
absolute invariant drifts and the raw residual expose the failure.
Repeating the repaired trajectory at `rtol=1e-9` and `1e-11` gives
matching statuses, four ordered horizon events and converging end
affine parameter/final state (the corresponding regression test bounds
event/end-time differences by `2e-7`). DOP853 remains in use.

## Regular and super-extremal examples

Default settings unless otherwise specified: `rtol=1e-10`, coordinate
time `atol=1e-11`, each other evolved component `atol=1e-12`,
`max_step=0.5`, `epsilon_sigma=1e-4`, `escape_radius=8`, and
`max_affine_parameter=35` in the report script. The threshold
`1e-8` for *each* scaled null, `L_z` and Carter diagnostic is fixed.

| Ray | Status; region changes | Min `|r|`; min `Sigma` | Max scaled null; absolute `L_z` / Carter drift | Steps; nfev |
| --- | --- | --- | --- | ---: |
| `a=.8`, `(0.4,.1,3)`, `d=(0,0,-1)` | `ESCAPED`; outer IN, inner IN, positive → negative disk | 5.40e-16; 0.529610 | 2.61e-13; 2.93e-14 / 1.76e-13 | 104; 1646 |
| `a=.8`, axis `(0,0,3)`, `d=(0,0,-1)` | `ESCAPED`; outer IN, inner IN, positive → negative disk | 3.47e-17; 0.640000 | 2.27e-16; 0 / 0 | 100; 1502 |
| `a=.8`, negative sheet `(0,0,-.1)`, `d=(0,0,1)` | `ESCAPED`; negative → positive disk, chart handoff, inner OUT, outer OUT | 3.47e-18; 0.640000 | 7.98e-11; 0 / 0 | 84; 1276 |
| `a=1.2`, `(1.3,.1,3)`, `d=(0,0,-1)` | `ESCAPED`; positive → negative → positive disk; **no horizon** | 5.56e-15; 0.227723 | 4.90e-11; 2.22e-11 / 2.03e-10 | 99; 1739 |
| `a=1.05`, `(1,.1,3)`, `d=(0,0,-1)` | `ESCAPED`; positive → negative disk; **no horizon** | 3.27e-16; 0.407101 | 4.09e-12; 7.23e-13 / 3.94e-12 | 62; 1112 |

All these trajectories continue across a disk with positive `Sigma`;
the equatorial `a=.7` plunge instead stops at the genuine ring
approach guard `Sigma=1e-4` without any disk event. The extremal
`a=1` axis example preserves one (nonterminal) outer-horizon event,
followed by a disk crossing and escape on the negative sheet.

## Packaging and remaining limits

`pyproject.toml` now uses the valid PEP 440 version **`0.1.0`**, with
a regression test parsing it via `packaging.version.Version`. An
isolated venv editable-install smoke used
`python -m pip install --no-deps --no-build-isolation -e .` and imported
the signed-sheet module; metadata reported `0.1.0`.

Each chart covers the horizon direction for which it is regular; a
failed safe chart handoff is reported as `CHART_BOUNDARY`. This model
does not assign all global Penrose blocks, and ray coordinates across
a handoff must be interpreted using `sample_charts`. For very close
super-extremal near-ring rays, default `rtol=1e-10` can exhaust the
`1e-8` conserved-quantity diagnostic bound while `Sigma` remains
nonzero. Example `a=1.05`, `(1.15,0.1,3)`, local direction `(0,0,-1)`:
`DIAGNOSTIC_FAILURE` at `lambda=4.507397` with Carter scaled drift
`1.01e-8`; at `rtol=1e-11` it reaches `ESCAPED`, with scaled Carter
drift `8.91e-10`. This is a measured tolerance-sensitive limitation,
and an early diagnostic failure is **not** presented as a physical
termination. Threshold `Sigma=epsilon_sigma` marks an approach,
not exact ring contact. Sign-change event bracketing can miss an even
number of crossings within a step. Rejected-step counts are unavailable
from the public DOP853 API; they remain `None`. No renderer, UI,
camera, Kerr–Newman charge physics or live recomputation is included.

# v0.1F comprehensive headless validation

Executed in Work on 2026-09-20: Linux, Python 3.12.14, NumPy 2.3.5,
SciPy 1.17.0. From this package directory:

    PYTHONPATH=src python -m unittest discover -s tests -v
    PYTHONPATH=src python scripts/v0_1f_diagnostics.py --repeat 7 --json-output reports/v0_1f_diagnostics.json

**60/60 tests pass**: the accepted **48 unchanged v0.1A–E tests** plus
12 new v0.1F checks. Existing older tests were not modified in v0.1F.
The v0.1E checkpoint had already revised one original v0.1D test to
expect regular disk continuation; its other 33 original tests remained
unchanged. Package version `0.1.0` remains valid PEP 440; v0.1F is
the checkpoint label, not a Python distribution version.

## Exact extremal and two-sided spin boundary

At `a=1` the exact horizon has `r_+=r_-=1`. An outward exterior
launch from `(4,.3,.6)` escapes with no horizon events. An equatorial
inward ray from `(4,0,0)` reaches the finite ring-approach guard after
exactly one **outer** inward event. Inclined inward rays from
`(4,.5,.6)` and `(3,.2,1)` cross that one horizon inward, turn,
handoff ingoing→outgoing in the extremal interior, cross the same
horizon outward and escape. A near-axis launch from `(.15,.05,3)`
crosses the one horizon inward, traverses the regular disk to negative
`r`, and escapes. All five outcomes are `COMPLETE` under the fixed
`1e-8` scaled null/`L_z`/Carter guard; the maximum among their
individual scaled invariant diagnostics is `5.33e-11`. No artificial
`INNER_HORIZON` event appears at exact extremality.

The same inclined `RayDefinition([4,.5,.6],[-1,0,0])` was solved at
`a=0.95,0.99,0.999,0.9999,1,1.0001,1.001,1.01,1.05`.
All nine results are reliable `ESCAPED`. Horizon-event counts are 4
for `a<1`, 2 at exact `a=1` and 0 for `a>1`; each has one regular
chart handoff. At `a=.9999,1,1.0001` the respective end affine
parameters are `9.448100285,9.448081650,9.448063024`. Independent
equatorial inward and near-axis disk rays also pass this nine-spin
regime check: the equatorial ray approaches the ring with 2/1/0
nonterminal horizon events, while the near-axis ray escapes after a
regular positive→negative disk crossing with 2/1/0 events. A selected
negative-spin exterior launch at `a=-.999` escapes reliably.

On the **common affine interval `lambda in [0,1.2]` before chart
handoff**, actual integrated trajectories were resampled by cubic
Hermite interpolation with analytic Hamiltonian position derivatives.
The maximum Cartesian distance between `.9999→1` and `1→1.0001`
solutions is `8.29118e-6` and `8.29170e-6`, respectively. The
normalized differences per unit spin are `0.082912` and `0.082917`.
Resampling is only a numerical comparison of independently solved
trajectories; no new ray or intermediate-spin result is inferred from it.
Metric variation and the shared-chart trajectories are smooth here;
changes in horizon-event count reflect the physical spin regime.

## Fixed-launch capture→escape example and super-extremal sheets

The equatorial unit local direction `(-sqrt(.75),.5,0)` from `(4,0,0)`
gives `SINGULARITY_APPROACH` at `a=.95` (outer and inner horizon inward,
minimum `Sigma=1e-4`) but `ESCAPED` at `a=1.05` (no horizons and
minimum `Sigma>1`). Each spin independently rebuilds the observer,
covector and invariants: `E` changes `1.21401→1.21822` and `L_z`
`2.21044→2.23568`. The sub-extremal ray has
`P(r_-)=E(r_-²+a²)-a L_z<0`; the ingoing horizon patch is unsuitable
for that inner branch. Geometry selects a finite switch at
`r=(r_++r_-)/2`, makes the canonical outgoing-KS state handoff and
continues past the inner horizon without a physical termination.
Exact-extremal and super-extremal radial turns use the same two patch
machinery where mathematically valid; DOP853 is unchanged.
Tightening `rtol` from `1e-10` to `1e-11` preserves both fates, with
terminal affine differences `2.64e-10` and `1.46e-8`. At
`rtol=1e-9` the `a=1.05` ray gives `DIAGNOSTIC_FAILURE`, so that
looser tolerance is not a safe preview for this example.

At `a=1.0001,1.001,1.01`, the near-axis close ray from
`(.8,.1,3)` crosses the regular disk into negative `r` and escapes.
`a=1.05` from `(1,.1,3)` does likewise. The `a=1.2` ray from
`(1.3,.1,3)`, directed along local `-z`, has two disk crossings
positive→negative→positive and escapes with **zero** horizon events:
minimum `Sigma=0.227723` (versus the `1e-4` ring approach guard),
max scaled null `4.90e-11`, max scaled `L_z` drift `2.22e-11`,
Carter drift `1.38e-10`. Its tighter `rtol=1e-11` run preserves both
disk crossings and escape; final affine difference `1.67e-10`.
No regular disk was mislabeled as a singularity.

## Classification and reliability contract

`assess_trajectory` reports `COMPLETE` only for a trusted finite
`ESCAPED` or `SINGULARITY_APPROACH` event. `MAX_INTEGRATION_LIMIT`
and `CHART_BOUNDARY` are `PARTIAL`: their sampled prefix has no final
physical fate. `NUMERICAL_ERROR`, `DIAGNOSTIC_FAILURE`, inconsistent
samples, or scaled null/`L_z`/Carter residuals above `1e-8` are
`UNRELIABLE`, even if a nominal terminal label was present. The
threshold is the unchanged empirical rejection guard from v0.1E,
not a theorem about global integration error. `p_t` and therefore
`E` are held exact by construction, so they are not independent drift
tests. Classifications near a capture boundary also require tolerance
convergence. Event bracketing can miss paired crossings within one
accepted step; a finite `Sigma` guard does not prove exact ring contact.

## Sequential CPU baseline and profile

Two updates at `a=.9999` and `a=1.2` use the same bank of ten
RayDefinitions. It includes exterior, horizon crossing, inclined,
near-axis, disk, equatorial capture/scatter and longer inner/double
disk rays. Seven timings per batch, after warmup, use normal accuracy
`rtol=1e-10`, `atol=(1e-11,1e-12×6)`, `max_step=.5`, escape radius 8.
All measured batch ray results were `COMPLETE`. The empirical p95
from seven observations is only a rough guide on this shared CPU.

| Spin | Rays | Median batch (s) | Empirical p95 (s) | Total seven solves (s) | Median per ray (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| .9999 | 1 | .0923 | .1117 | .6807 | .0923 |
| .9999 | 2 | .5791 | .7363 | 4.1867 | .2895 |
| .9999 | 5 | 1.5643 | 2.1323 | 11.4152 | .3129 |
| .9999 | 10 | 3.3141 | 3.6065 | 22.2163 | .3314 |
| 1.2 | 1 | .0650 | .0670 | .4492 | .0650 |
| 1.2 | 2 | .5489 | .6149 | 3.7796 | .2744 |
| 1.2 | 5 | 1.1842 | 1.5606 | 8.8111 | .2368 |
| 1.2 | 10 | 3.7573 | 4.6300 | 26.3573 | .3757 |

A five-ray `a=.9999` cProfile sample spent **self time** approximately
47.7% in geometry/Hamiltonian, 25.9% in NumPy, 19.6% in other Python
and native calls, 3.9% in SciPy solver, 2.8% in integrator/events,
0.1% in tetrad launch. `null_geodesic_rhs` appeared 6171 times,
showing metric/RHS numerical evaluation dominates repeated batch work;
frame creation and chart switching are minor here. Instrumentation
changes timings; the table above is the unprofiled baseline. On this
Work CPU, ~3–4s for 10 rays is unsuitable for continuous synchronous
slider recomputation. A future preview/refinement UI needs explicit
scheduling and accuracy validation; the tested `rtol=1e-9` is unsafe
for the capture→escape example. No preview solver, parallelism, GPU,
UI or renderer was implemented. Full observations and event data are
in `reports/v0_1f_diagnostics.json`.

## Packaging, remaining limits and Windows validation debt

The present extension supports the tested radial branches, not every
block of the maximal analytic extension. Other near-ring rays can hit
the unchanged invariant guard, at which point their later trajectory
and apparent fate must not be treated as physics. Sampled minimum
`Sigma`/`|r|` is not necessarily a global continuous minimum.
SciPy's public DOP853 does not expose a reliable rejected-step count.

**Validation debt:** Neither v0.1E nor v0.1F has yet been manually
validated on the user's Windows 11 / Python 3.14.4 / NumPy 2.5.3 /
SciPy 1.18.1 machine. From the unpacked v0.1F checkpoint directory,
when that machine is available, run:

    py -m pip install -e .
    py -m unittest discover -s tests -v
    py scripts/v0_1f_diagnostics.py --repeat 3

The test command should report 60 passing tests. Omitting `--repeat 3`
runs the same seven-timing benchmark as above, which is more useful
for local p95 estimates. Compare ray statuses and invariant gates as
well as elapsed times; these Work CPU timings do not predict Windows
latency.

Editable-install smoke in a **separate virtual environment** completed:
`python -m pip install --no-deps --no-build-isolation -e .` produced
editable wheel `kerr_geodesics_geometry-0.1.0-0.editable-py3-none-any.whl`.
Importing `kerr_geodesics` from a directory outside the project resolved
to this v0.1F source tree. Installed metadata parsed as PEP 440 `0.1.0`;
an exact-extremal outward solve returned reliable `ESCAPED`.
The smoke virtual environment used system site packages for the installed
NumPy/SciPy; it is not bundled with the checkpoint. The existing
project-version regression test also remains passing. The final zip
is separately tested for CRC/readability when built; its SHA-256 is
reported alongside the download link.

# v0.2A headless CPU compute validation

**Prior validation debt closed:** the user ran the complete v0.1F
60/60 suite on Windows 11, Python 3.14.4, NumPy 2.5.3 and SciPy
1.18.1, confirming signed-r, chart handoffs, extremal/super-extremal
and the tighter-tolerance capture→escape fixture. The historical
v0.1E/v0.1F Windows debt above is therefore resolved. This new v0.2A
compiled backend and process scheduling still require their own run
on that Windows machine.

Work measurements (Linux, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0)
use **exactly the ordered ten-ray v0.1F bank**, `escape_radius=8`,
`rtol=1e-10`, the original seven absolute tolerances and `max_step=.5`.
Two spin updates `.9999` and `1.2`, seven repetitions per batch, after
warmup; empirical p95 from seven runs is noisy. Neither accuracy nor
event/invariant rejection thresholds changed. Baseline was recorded
**before** modifying v0.1F physics or integrating the compiled kernel:

    PYTHONPATH=src python scripts/v0_2a_benchmark.py --phase baseline --repeat 7 --json-output reports/v0_2a_baseline.json

After installing the optional Numba extra, the new backend was tested:

    PYTHONPATH=src python scripts/v0_2a_benchmark.py --phase optimized --repeat 7 --workers 4 --json-output reports/v0_2a_final.json

| Spin | Rays | v0.1F reference median / p95 (ms) | Compiled sequential median / p95 (ms) | Warm process-4 median / p95 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| .9999 | 1 | 54.2 / 64.7 | 8.9 / 9.7 | 14.1 / 17.2 |
| .9999 | 2 | 341.8 / 374.0 | 55.8 / 61.6 | 60.2 / 85.8 |
| .9999 | 5 | 856.8 / 888.6 | 141.9 / 148.1 | 87.5 / 91.1 |
| .9999 | 10 | 1660.4 / 1794.2 | 305.8 / 307.9 | 149.9 / 184.1 |
| 1.2 | 1 | 42.6 / 43.4 | 8.1 / 8.6 | 10.7 / 11.7 |
| 1.2 | 2 | 325.6 / 333.9 | 51.7 / 53.7 | 51.0 / 54.0 |
| 1.2 | 5 | 742.9 / 792.6 | 115.7 / 119.3 | 51.3 / 69.9 |
| 1.2 | 10 | 1359.7 / 1415.6 | 235.2 / 264.7 | 108.3 / 117.5 |

The compiled path is roughly 4.7–6.3× faster sequentially for these
batches. Four warmed processes reduce the 10-ray median to roughly
`150/108 ms` in the final seven-repeat run, but have small overhead
for a single easy ray; use the
sequential compiled path for one ray. Seven-batch totals and individual
timing observations are preserved in the two JSON reports. CPU and
SciPy version differ from the user's machine, so **these numbers do
not predict Windows latency**.

The one-ray row is the **first, easy exterior ray**, not a bound on
all single-ray difficulty. Separate warmed single-ray medians/p95
are: inclined inner-region return at `.9999`: `38.6/44.6 ms`;
equatorial `.95` capture with inner-horizon chart selection:
`56.4/75.8 ms`; super-extremal `1.2` double-disk path:
`44.7/48.2 ms`. The 20–30 ms objective is reached for easy rays
but **not** for these difficult ones.

### Why this specific optimization

The pre-change five-ray cProfile sample at `a=.9999` reported:

| Reference function | Calls | Instrumented cumulative seconds |
| --- | ---: | ---: |
| `radial_coordinate` | 24,234 | .162 |
| `domain` | 20,807 | .380 |
| `inverse_metric` | 6,483 | .244 |
| `inverse_metric_derivatives` | 6,171 | .710 |
| `null_geodesic_rhs` | 6,171 | 1.043 |
| integrator `rhs` callback | 5,697 | 1.094 |
| event `measure` | 2,836 | .087 |
| diagnostic `record` | 306 | .041 |
| chart handoff | 1 | .0002 |

A compiled scalar proof-of-concept evaluated the existing analytic
inverse-metric and derivative contraction in approximately **2.4 µs**
per point versus **102 µs** for the reference NumPy-heavy callback.
Six positive/negative, ingoing/outgoing, disk-hemisphere,
Schwarzschild and even off-shell pointwise checks matched within
`5e-14` absolute/`2e-13` relative (observed typically `1e-16`).
The geometry now owns one fused kernel that reuses `r`, `Sigma`, ell,
and analytic derivative intermediates *within the same evaluation*.
No previous DOP853 evaluation's mutable data is cached. The public
pointwise reference metric/derivative/Hamiltonian API is unchanged.
The five-ray optimized profile puts `integrate_ray` at `.341s`,
SciPy `rk_step` at `.161s`, RHS callbacks at `.157s`, event `measure`
at `.043s` and diagnostics `record` at `.039s`; chart handoff still
about `.0002s`. These are overlapping **cumulative, instrumented**
times, not additive wall-clock components. The dominant remaining
work is Python/SciPy accepted-step and event/diagnostic orchestration,
not tetrad initialization or one chart handoff.

Numba 0.67.0 supports the target Python 3.14 / NumPy 2.5 according
to its [official compatibility table](https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information),
but this checkpoint has **not** run on Windows yet. The optional
`performance` extra is `numba>=0.67,<0.68`; plain editable installs
retain the reference backend. Measured first uncached kernel compile
in this Work environment was `.557s`; a subsequent process loading
cached code spent `.148s` in explicit warmup. Spawning and warming four
processes for the final run cost `.844s` before warmed timing. Import
time and Windows process-start latency were not separately measured;
never mix them with the warmed medians in the table.

### Worker choice, equivalence and scheduler

An initial three-repeat worker survey measured both thread and
spawn-process options, counts 2/4/6. For 10 rays at `a=.9999`, thread
medians were `.505/.622/.655s`: all **slower** than `.314s` compiled
sequential. The production ray-level thread pool was removed. The
process medians were `.176/.112/.102s` with 2/4/6 workers;
at `a=1.2` they were `.131/.102/.083s`. Four is the bounded default:
near six-worker performance for 10 rays without allocating one worker
per ray. Six remains a tunable measured option; rerun on the user's
24-logical-CPU machine before raising the worker count. This survey is
preserved in `reports/v0_2a_parallel_probe.json`.

The original **60 tests remain unchanged and passing**; seven new
backend/compute tests bring the total to **67/67** with Numba
installed. Test cases separately compare reference and compiled RHS
on both signed sheets/charts, nine representative full trajectories,
the Schwarzschild photon sphere, an explicitly cancelled DOP853
solve, ordered process results, request-ID isolation and cooperative
latest-request-wins cancellation. The final diagnostic additionally
compared reference, sequential compiled and four-worker process
solutions across **49 spin/ray combinations**, including both sides
of the fixed-definition capture→escape fixture: all 49 matched fate,
reliability, horizon/branch/chart transitions and final positions.
Maximum differences were `2.2061e-10` in end affine parameter and
`6.7851e-9` in final Cartesian position components. These 49 had
38 reliable escapes and 11 reliable finite singularity approaches.
In a reference
environment without Numba, 67 tests pass with the four
compiled/process-specific tests skipped; all original 60 still run.

The scheduler demonstration submitted `.95→.96→.97` with one active
and one queued stale request. Their compute statuses were
`SUPERSEDED`, `SUPERSEDED`, `DONE`; no stale result was publishable.
The final result retained its `ray_id`, and trajectories are read-only
snapshots. Cancellation is raised *outside* geodesic
`TerminationStatus`, never disguised as `ESCAPED`, `NUMERICAL_ERROR`
or another physics outcome. `DONE` denotes compute completion, not
physical reliability; assess each ray independently. The scheduler
currently uses cancellable **sequential** optimized solves; the fast
spawn-process batch pool is separately reusable and warmed. Integrating
cooperative cross-process cancellation into that pool is still open.
Progressive/partial trajectory publication was deliberately deferred;
no lower-accuracy preview path was added.

### Packaging and outstanding v0.2A Windows validation

The valid PEP 440 project version remains `0.1.0` (not the checkpoint
label). This Work environment needs a Windows-target check specifically
for Numba wheels/JIT output, spawn-process startup, parallel timing and
scheduler cancellation. From the unpacked checkpoint root on the
user's previously verified Windows 11 / Python 3.14.4 / NumPy 2.5.3 /
SciPy 1.18.1 machine, run:

    py -m pip install -e ".[performance]"
    py -m unittest discover -s tests -v
    py scripts/v0_2a_benchmark.py --phase optimized --repeat 7 --workers 4

The test suite should report **67 tests** with no skip when Numba is
installed. Optionally rerun `py scripts/v0_2a_benchmark.py --phase
baseline --repeat 7` to compare against the user's own v0.1F
sequential CPU times. Absolute live-slider readiness cannot be
certified from Linux results alone; 10 rays around `100 ms` is
promising for asynchronous updates, **not** 60-Hz full solves.
Full benchmark logs, test count, equivalence and scheduler outcomes
are in `reports/v0_2a_final.json`.

An isolated Python 3.12 virtual environment with installed NumPy/SciPy
completed `python -m pip install --no-build-isolation -e ".[performance]"`;
metadata reported valid PEP 440 `0.1.0`, with
Numba `0.67.0` and llvmlite `0.49.0` resolved. Importing from *outside*
the project used this v0.2A editable source; explicit JIT warmup and
an exact-extremal compiled solve returned reliable `ESCAPED`.
The final archive is CRC-checked independently; its hash appears with
the download link. No virtual environment or generated bytecode is
included in the checkpoint.

## v0.3A static viewer: Work checks and Windows gate

The user's Windows 11 / Python 3.14.4 v0.2A checkpoint passed 67/67 and
49 representative physics equivalences. No physics equations, solver,
reliability policy or integration tolerance was altered in v0.3A. The sole
geometry addition is `canonical_ingoing_position`, which undoes the
established outgoing-chart azimuthal coordinate rotation **for display**.
The old 67 test files remain unchanged. Six new headless viewer tests bring
the Work test result to **73/73** with the optional Numba dependency.

Commands run on Linux Python 3.12.14 (isolated editable installation):

    python -m unittest discover -s tests -q
    python scripts/v0_3a_smoke.py --egl --json-output reports/v0_3a_smoke.json

Five scenarios (`a=0,0.7,1,1.05,1.2`) generated expected horizon
visibility and red point/ring. All 15 actual physics rays were COMPLETE;
regular signed-r disk crossings and chart transitions remained visible.
ModernGL 3.3 shaders compiled and rendered to a 640×480 EGL Mesa/llvmpipe
software framebuffer in all five cases with non-background pixels and an
opaque compositing alpha channel. The ray projection is pixel-width GPU
ribbon triangles; alpha-blended horizon is drawn before always-visible rays.
`reports/v0_3a_egl_preview.png` is a saved 0.95 software-rendered scene.
The checked EGL image is not a Qt window or proof of interactive camera,
resize/DPI behavior or 60 FPS on the RTX 5060.

`QT_QPA_PLATFORM=offscreen` here reports `QOpenGLWidget is not supported on
this platform` and fails to create a Qt OpenGL context. Thus **real Qt UI,
mouse/keyboard input, Qt FBO binding, Windows DPI and GPU FPS are unverified**.
Windows local commands after extraction:

    py -m pip install -e ".[viewer-performance]"
    py -m unittest discover -s tests -v
    py scripts/v0_3a_viewer.py --spin 0.95
    py scripts/v0_3a_viewer.py --spin 0
    py scripts/v0_3a_viewer.py --spin 0.7
    py scripts/v0_3a_viewer.py --spin 1
    py scripts/v0_3a_viewer.py --spin 1.05
    py scripts/v0_3a_viewer.py --spin 1.2

**Packaging blocker:** PyPI supplies PySide6 6.11.2 Windows CPython abi3
wheels, but its latest ModernGL 5.12.0 and glcontext 3.0.0 have no CPython
3.14 Windows binary wheels. A pip binary-only resolution check for win_amd64
cp314 returned `No matching distribution found` for *both*. A source build
needs MSVC and has not been validated on Python 3.14. Do not count Windows
viewer installation, interactivity or the 60 FPS goal as passed until the
user actually verifies them. PEP 440 package version stays `0.1.0`.

## v0.3A-r1 ray readability patch — current Windows target

The user confirmed the v0.3A Qt viewer, camera and basic horizon/
singularity visualization work locally. The *official new target* is
Windows 11 / Python **3.13.x** / NumPy 2.5.x / SciPy 1.18.x /
Numba 0.67.x / PySide6 / ModernGL. Earlier Python 3.14 viewer
packaging warnings above are historical and no longer a checkpoint gate.
A binary-only win_amd64 CPython 3.13 resolver probe successfully fetched
ModernGL 5.12.0, glcontext 3.0.0 and PySide6 6.11.2 wheels. Python 3.13
viewer interactivity and image quality for the *r1 markers* still need
user-side checking; this Linux Work environment has no Qt desktop context.

This patch changes only the `kerr_viewer` package, its tests and docs.
`kerr_geodesics` files, all original three ray positions/directions,
DOP853 settings, termination semantics, compiled backend and scheduler
are byte-for-byte unchanged from v0.3A. Default ray labels are now
Exterior escape (yellow), Inclined horizon return (cyan), Signed-r disk
passage (magenta); these names do not alter launch data.
The 73 existing tests are unchanged; seven new headless readability
tests bring the Work result to **80/80 passed** with Numba enabled.
An isolated Linux Python 3.12 editable install with the viewer-performance
extra succeeded at PEP 440 project version `0.1.0`, PySide6 `6.11.2`
and ModernGL `5.12.0`. Python 3.13 Windows installation is still
awaiting the user's local run.

For each ray the START glyph is placed at the first *recorded* Cartesian
sample and the fate-specific END glyph at its last *recorded* sample.
Four camera-facing direction arrows use existing successive samples;
there is no spline, resampling or change to trajectory position arrays.
Horizon event positions come directly from recorded physics event state,
converted by the existing geometry chart mapper; disk/chart events use
exact matching trajectory lambda samples when recorded. If a non-chart-
crossing event lies between saved samples, only its marker is interpolated
linearly from saved positions in the same chart; no event is attached to
an arbitrary nearest sample, and no missing cross-chart event position is
invented. Panel labels inherit physics INWARD/OUTWARD classification.

The EGL software OpenGL smoke and 0.95 image preview are stored under
`reports/v0_3a_r1_smoke.json` and `reports/v0_3a_r1_egl_preview.png`.
They verify shader compilation and offscreen marker pixels, not Windows
Qt behavior or performance. Renderer color/glyph policy and event placement
have dedicated headless tests. **Local Windows r1 check:**

    py -3.13 -m pip install -e ".[viewer-performance]"
    py -3.13 -m unittest discover -s tests -v
    py -3.13 scripts/v0_3a_viewer.py --spin 0.95
    py -3.13 scripts/v0_3a_viewer.py --spin 0.7
    py -3.13 scripts/v0_3a_viewer.py --spin 1.0
    py -3.13 scripts/v0_3a_viewer.py --spin 1.05
    py -3.13 scripts/v0_3a_viewer.py --spin 1.2

The default 0.95 scene should show a yellow exterior escape, cyan horizon
return with OUTER-I/INNER-I/CHART/INNER-O/OUTER-O, and magenta signed-r
disk passage with OUTER-I/INNER-I/DISK. All three in the verified Work
solve have ESCAPED/COMPLETE. Window interaction and FPS must still be
checked by the user on Windows.
