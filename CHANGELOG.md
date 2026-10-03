# Changelog

## v0.2A (Live Compute Performance Core)

Added an optional Numba 0.67 fused scalar CPU callback owned by the
Kerr-Schild geometry. The public metric/derivative/Hamiltonian reference
path, DOP853, signed-r and chart continuation, reliability thresholds
and launch convention remain unchanged. One RHS evaluation reuses its
radial/metric/derivative intermediates; no cross-state caching exists.
Added explicit JIT warmup, a persistent ordered spawn-safe process
batch executor, headless latest-request-wins request generations and
compute-level cancellation statuses. Measured and discarded ray-level
threads after they made batches slower. Added physics equivalence,
parallel ordering, scheduler and cancellation regressions and a
reproducible baseline/optimized/parallel benchmark. The v0.1F Windows
validation debt is closed by the user's 60/60 local run; v0.2A has
not yet been tested on that machine. Package version remains the
PEP 440 string `0.1.0`, independent of checkpoint name. No UI,
renderer, CUDA or Kerr–Newman charge was added.

## v0.1F (Comprehensive Physics Validation + Spin-Sweep Readiness)

Added headless sequential same-definition spin sweep, one-spin batch
solves and explicit trajectory reliability assessment. Validated five
exact-extremal launches, both sides of the near-extremal boundary,
super-extremal regular disk and two-way sheet crossings, an equatorial
capture-to-escape example, and tolerance tightening. Geometry now
selects the outgoing KS patch for otherwise incompatible ingoing
inner-horizon branches; exact extremal and super-extremal regular
radial returns can also switch patches. No solver, metric, initial
launch convention, or reliability threshold was changed. Added
common-affine trajectory comparisons, an executable 1/2/5/10-ray CPU
benchmark and profiler, and a recorded Windows local validation debt.
The PEP 440 distribution version remains `0.1.0`; the checkpoint
label is v0.1F. No renderer, UI, scheduler or charge physics was added.

## v0.1E (Signed-r Extension + Inner-Region Robustness)

Added explicit signed Kerr sheets, regular disk hemisphere continuation,
analytic metric/derivative/Hamiltonian evaluation on negative r, and
ingoing/outgoing horizon-penetrating KS chart handoffs with canonical
momentum transformation. Trajectories record per-sample sheet/chart/
radial region and ordered disk/horizon/chart transitions. Introduced a
`DIAGNOSTIC_FAILURE` guard for scaled null, L_z or Carter drift beyond
`1e-8`. The formerly failing inclined inner-horizon ray now reaches the
escape boundary with controlled invariants. Added negative-sheet FD,
chart-Jacobian, regular-disk/ring, extremal and super-extremal tests.
Fixed the project metadata version to PEP 440 `0.1.0` and checked an
editable installation. One previous regular-disk chart-boundary test
was updated because signed-r continuation changes its correct physical
outcome. No renderer or UI was added.

## v0.1D (DOP853 Single-Ray Integrator and Events)

Added forward-affine DOP853 integration of one null ray, fixing p_t by
stationarity. Dense-output roots distinguish outgoing finite-radius ESCAPED,
Sigma-threshold SINGULARITY_APPROACH, unsupported regular r=0 CHART_BOUNDARY,
MAX_INTEGRATION_LIMIT and NUMERICAL_ERROR. Outer and distinct inner horizons
are nonterminal recorded crossings. Added raw and scaled null residuals,
conserved-quantity drift, step diagnostics, a real-ray diagnostic script and
regression tests. No renderer, UI or chart extension exists.

## v0.1C (Local Observer Frame and Null Launch)

Added the Kerr-Schild time-slice Eulerian observer, symmetric inverse-square-
root spatial triad, reusable local ray definition, null tangent/covector
construction, and geometry-owned E, L_z and carter_constant evaluation.
Added numerical frame checks across both sides of horizons. The 15 earlier
tests remain unchanged.

## v0.1B (Metric Derivatives + Hamiltonian RHS)

Added analytic inverse-metric derivatives to the geometry interface and Kerr
implementation, plus a geometry-agnostic pointwise Hamiltonian and null-geodesic
RHS. Added independent test-only central-difference comparisons and a fixed
validation report. The original seven v0.1A tests are unchanged. No
integrator, trajectory, tetrad, UI, or Kerr-Newman equations are included.

## v0.1A (Geometry Core)

Independent positive-r Cartesian Kerr-Schild metric implementation,
immutable parameters with Q_bh field, singularity/domain and horizon
metadata, unit tests, and a reproducible numerical identity report.
The v0.1A package is the working baseline before integration work.

## v0.3A (Basic static 3D viewer)

Added optional PySide6 window and ModernGL 3.3 renderer consuming static
read-only physics trajectory snapshots; 3D fly/orbit camera, horizon spheroid,
red singularity marker, z axis, pixel-width ray ribbons and diagnostics.
Added display-only outgoing→ingoing Cartesian chart mapping to geometry.
No changes to numerical physics or scheduler and no interactive spin slider,
ray launch or crosshair. Six new headless tests; detailed platform caveat in
VALIDATION.md. Package version remains valid PEP 440 `0.1.0`.

## v0.3A-r1 (Ray Readability / Visual Debug)

Retained the exact v0.3A physics and ray fixtures. Added deterministic
viewer-only yellow/cyan/magenta palette, start/fate endpoint glyphs,
four sparse direction arrows and horizon/disk/chart event glyphs at
recorded/interpolated event locations, plus panel event directions,
finite escape-boundary explanation and COMPLETE/PARTIAL/UNRELIABLE
styles. Toggle marker overlays with M. Python 3.13 Windows wheel
availability was checked; Windows GUI visual verification remains local.
The Python package metadata stays valid PEP 440 `0.1.0`.
