# Kerr null-geodesic simulator, v0.3A-r1 static viewer

This is a standalone headless Kerr geometry, pointwise Hamiltonian RHS,
local-frame null launch, affine geodesic integrator, and sequential headless
spin-sweep and batch solving APIs, plus an optional compiled CPU callback,
ordered process workers and headless latest-request-wins coordinator.
The optional `kerr_viewer` package adds a static PySide6/ModernGL viewer.
The physics package imports without either viewer dependency.
Its source is independent of the user's older ray-tracing application.

## Install and validate

From this directory on Windows (Python 3.11 or newer):

    py -m pip install -e .
    py -m unittest discover -s tests -v
    py scripts/geometry_diagnostics.py
    py scripts/v0_1b_diagnostics.py
    py scripts/v0_1c_diagnostics.py
    py scripts/v0_1d_diagnostics.py
    py scripts/v0_1e_diagnostics.py
    py scripts/v0_1f_diagnostics.py
    py scripts/v0_2a_benchmark.py --phase baseline

For the optimized CPU backend, install the optional Numba dependency
instead of the plain editable install and pre-warm before live requests:

    py -m pip install -e ".[performance]"
    py scripts/v0_2a_benchmark.py --phase optimized --workers 4

On systems with python rather than py, substitute python. NumPy and SciPy are required.
The units and coordinate signature are G=c=M=1 and (-,+,+,+), respectively.
Coordinates are ordered (t,x,y,z); the spin axis is +z. Signed a_star is
permitted. The data structure reserves Q_bh for a later model; this Kerr
implementation rejects Q_bh != 0 explicitly. A ray's Carter integral is
always named `carter_constant` (distinct from `Q_bh`).

## Geometry and sources

On the original positive-r ingoing Cartesian Kerr-Schild chart define

    b = x*x + y*y + z*z - a*a
    r*r = (b + sqrt(b*b + 4*a*a*z*z))/2
    Sigma = r*r + a*a*z*z/(r*r)
    H = r/Sigma
    ell_mu = (1, (r*x+a*y)/(r*r+a*a),
                 (r*y-a*x)/(r*r+a*a), z/r)
    g_mu_nu = eta_mu_nu + 2*H*ell_mu*ell_nu
    g^mu_nu = eta^mu_nu - 2*H*ell^mu*ell^nu

In v0.1E the magnitude of this root is combined with a signed sheet
`branch=+1/-1`, and the ingoing/outgoing chart is selected separately.
Across a regular disk the sign of `r` changes while `cos(theta)=z/r`
remains continuous. At the exact disk the required hemisphere selects its
finite limit. For outgoing KS, the radial parts of `ell_i` reverse sign;
the terms with `a` retain the same sign. The square root for b<0 is
evaluated in rationalized form to avoid
subtracting nearly equal numbers near the r=0 disk. Raising ell here
uses eta, as allowed by the Kerr-Schild null-vector identity. No inverse
metric is computed by numerically inverting g.

Primary formulas: SpECTRE's Kerr-Schild analytic solution:
https://spectre-code.org/classgr_1_1Solutions_1_1KerrSchild.html

Horizon and singularity reference: M. Visser, The Kerr spacetime:
https://arxiv.org/pdf/0706.0622

Horizon roots are r_+=1+sqrt(1-a*a) and r_-=1-sqrt(1-a*a) for
|a|<=1. The implementation uses equivalent arithmetic that behaves
better near |a|=0 and |a|=1. At a=0 the algebraic inner root r_-=0
is a singularity, not a distinct inner horizon. At |a|=1 the two
roots coincide. At |a|>1 there are no real horizon roots.

For a != 0 the exact singular set is the ring r=0, z=0,
x*x+y*y=a*a. The positive-root Cartesian formula also maps the
interior of that ring on z=0 to r=0. In v0.1A the regular disk was
a chart boundary. In v0.1E it is traversable with hemisphere data:
`Sigma=a*a-x*x-y*y>0` and the metric is Minkowski at `r=0` there.
An isolated exact-disk query without hemisphere is still ambiguous and
raises CoordinateDomainError, while a ray approaching it receives the
continuous hemisphere automatically. Domain classification
checks the exact represented sets, without an arbitrary-radius
numerical collision tolerance.

## Validation scope

The seven unchanged v0.1A tests verify the independent oblate-spheroidal equation for r, the
null Kerr-Schild vector, the rank-one metric identity, metric symmetry,
metric times inverse, determinant -1, known Schwarzschild metric
components, finite metric exactly at both horizons, and domain/horizon
regimes including |a|>1. The diagnostic script reports the maximum
absolute errors over a fixed seed and 1000 samples.

At the v0.1A checkpoint, geodesic initial conditions, Carter integral,
integration across a horizon and trajectory conservation had not yet been
implemented. The sections below document the newer stages.

## v0.1B: analytic inverse-metric derivatives

`Geometry.inverse_metric_derivatives(position)` returns `D[sigma, mu, nu]`
where `D = partial_sigma g^(mu nu)` and ordering is `(t,x,y,z)`.
For stationary Kerr, `D[0]` contains exact zeros.

Implicit differentiation of the positive-r defining equation gives

    r^4 - b*r^2 - a^2*z^2 = 0,     b = x^2+y^2+z^2-a^2
    Sigma = r^2+a^2*z^2/r^2 = 2*r^2-b
    partial_i r = (r*x_i + (a^2*z/r)*delta_iz)/Sigma.

Production code differentiates `H=r/Sigma` and each `ell_mu` analytically
with product and quotient rules. With `ell^mu=eta^(mu nu)ell_nu`, it computes

    partial_i g^(mu nu) = -2 * [
        (partial_i H)*ell^mu*ell^nu +
        H*((partial_i ell^mu)*ell^nu + ell^mu*(partial_i ell^nu))
    ].

The Kerr-Schild inverse and radial identities come from the
[SpECTRE Kerr-Schild documentation](https://spectre-code.org/classgr_1_1Solutions_1_1KerrSchild.html),
which lists an algebraically equivalent radial derivative. No metric
matrix is numerically inverted or finite-differenced in production.
Near the ring or unsupported r=0 disk crossings, v0.1B claims no
general numerical accuracy; it inherits v0.1A's positive-r domain.

## Pointwise Hamiltonian API

`hamiltonian_value(geometry, state)` evaluates
`H_geo=0.5*g^(mu nu)*p_mu*p_nu`. `null_geodesic_rhs(geometry, state)`
returns the derivative of one eight-entry state:

    state = (t, x, y, z, p_t, p_x, p_y, p_z)
    dx^mu/dlambda = g^(mu nu)*p_nu
    dp_mu/dlambda = -0.5*(partial_mu g^(alpha beta))*p_alpha*p_beta.

`p_mu` is **covariant**; its contraction with the inverse metric yields
contravariant `dx^mu/dlambda`. A null state satisfies `H_geo=0`.
For stationary Kerr `dp_t/dlambda=0` exactly. The Hamiltonian layer calls
only the `Geometry` interface, with no Kerr-specific formulas. The
canonical Hamilton equations here use an affine parameter; see also
[White, *Blacklight*, section 2.2](https://arxiv.org/abs/2203.15963).
The name `carter_constant` remains reserved for later per-ray diagnostics;
v0.1B does not compute any conserved quantity or trajectory.

## Independent numerical checks

All finite differences live in `tests/reference_calculations.py`. The
fixed test set covers `a_star=0, 0.7, -0.8, 0.999`, seven positions per
spin away from the singular/chart boundary, and two null momenta per
position. Test-only null covectors solve the metric's quadratic equation
for `p_t` with the future-directed root; no tetrad is constructed.
For each Cartesian coordinate `q`, central differences use
`h_q=1e-5*max(1,abs(q))` in `G=c=M=1` units, with covariant momenta held
fixed for the Hamiltonian gradient. The central scheme has O(h^2)
truncation error and floating-point cancellation roughly O(epsilon/h).
At these tested scales this h balances the two. Both maximum absolute
error acceptance limits are `2e-8`; these allow room for different
coordinate scales and are not guarantees near the singular set. The
null-Hamiltonian limit is `2e-15`; `dp_t/dlambda` must be exactly zero.
See `VALIDATION.md` for measured errors and the scope of the tests.

## v0.1C: local Eulerian observer and null launch

`RayDefinition(coordinate_position, local_direction)` stores the same
Cartesian position and **unit local** direction for reuse at different
stationary spin parameters; time defaults to 0 and `omega_local` is fixed
at 1. No physical camera motion is implied. `create_initial_conditions`
rebuilds all spin-dependent quantities from this definition. In particular,
it never reuses another spin's `p_mu`, `E`, `L_z` or `carter_constant`.

On supported positive-r slices, the Eulerian observer normal is
`n_mu=(-alpha,0,0,0)`, with `alpha=1/sqrt(-g^tt)` and
`n^mu=g^(mu nu)n_nu`. The spatial metric is `gamma_ij=g_ij`.
The triad columns of the unique positive symmetric matrix
`gamma^(-1/2)` have zero time component. This avoids arbitrary signs
of individual eigensystem vectors and maintains a Cartesian-aligned
frame through repeated spatial eigenvalues. Given a local unit direction,
`k^mu=n^mu+d^hat_i e_hat_i^mu` and `p_mu=g_mu_nu k^nu`. Tests independently
check `g(k,k)=0`, `g^(-1)(p,p)=0`, `-p_mu n^mu=1`, future orientation and
the recovered local direction. The lapse/spatial-metric relations also
appear in the [SpECTRE Kerr-Schild 3+1 quantities](https://spectre-code.org/classgr_1_1Solutions_1_1KerrSchild.html).

The geometry calculates `E=-p_t`, `L_z=x*p_y-y*p_x` and the **null**
`carter_constant`, where

    carter_constant = p_theta^2 + cos(theta)^2 *
        (L_z^2/sin(theta)^2 - a^2*E^2).

Our naming uses `Q=K-(L_z-aE)^2`, based on the alternate `K` convention in
[Bakun et al., Eq. 9b](https://arxiv.org/html/2409.03722v1).
The Cartesian implementation expands `p_theta^2+L_z^2*cot(theta)^2`
algebraically so the expression remains well-defined on the axis.
The test-only reference evaluates the separated formula away from the
axis and the Schwarzschild identity `carter_constant=|x cross p|^2-L_z^2`.
The same Eulerian time-slice frame is regular on the tested positive-r
points inside and outside horizons. Signed-r disk continuations are not
supported.

## v0.1D: one null ray with DOP853

    from kerr_geodesics import (
        SpacetimeParameters, KerrSchildGeometry, RayDefinition,
        IntegrationSettings, integrate_ray,
    )
    geometry = KerrSchildGeometry(SpacetimeParameters(a_star=0.7))
    ray = RayDefinition([4, 0, 0], [-1, 0, 0])
    result = integrate_ray(geometry, ray, IntegrationSettings(escape_radius=8))
    print(result.status, result.horizon_crossings)

SciPy [DOP853](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.html)
evolves `(t,x,y,z,p_x,p_y,p_z)` against **affine parameter**, restoring
the initial `p_t` exactly for each RHS call. The public result states have
the full eight components `(t,x,y,z,p_t,p_x,p_y,p_z)`. Defaults are
`rtol=1e-10`, `atol=(1e-11,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12)`,
`max_step=0.5`, and a finite affine/step budget. The accepted-step limit
is enforced by advancing DOP853 one accepted step at a time; its dense
interpolant and Brent bracketing locate sign-changing event roots.
The maximum step is also restricted by the *analytic* `dr/dlambda` as
`r` or `Sigma` approaches a termination threshold. No momentum projection
or finite-difference derivative is performed during integration.

| Status | Meaning |
| --- | --- |
| `ESCAPED` | `r=escape_radius` crossed outward; a finite boundary, not infinity. |
| `SINGULARITY_APPROACH` | `Sigma=epsilon_sigma` crossed inward; an approach threshold, not exact contact. |
| `CHART_BOUNDARY` | A required chart/domain continuation is unavailable at a safe finite patch handoff. |
| `MAX_INTEGRATION_LIMIT` | Finite affine or accepted-step budget ended without a physical termination. |
| `NUMERICAL_ERROR` | Integrator failure or evaluation exception; the message records the cause. |
| `DIAGNOSTIC_FAILURE` | A null, `L_z` or Carter reliability guard was exceeded; no physical fate is assigned. |

Outer and distinct inner horizon crossings are **recorded** with their
affine parameter, direction and eight-component state, and never stop
the ray. For `a=0`, the algebraic inner root `r_-=0` is not a regular
inner horizon. For `|a|>1` there are no horizon events.

Defaults `epsilon_sigma=1e-4` (M^2) and
`chart_boundary_radius=1e-3` (M) are numerical guards, not changes to
the metric or a visual collision radius. The first crossed threshold
determines the label. `minimum_sampled_r` and `minimum_sampled_sigma`
are minima of recorded states and events, not a claim of exact extrema
between samples. Event detection depends on a sign change across a
step; multiple crossings inside one step could be missed.

The result gives raw `max_abs_null_residual=|g^(mu nu)p_mu p_nu|` and
`max_scaled_null_residual`, dividing by
`max(1, ||g^(-1)||_infinity * ||p||_2^2)`. It also gives absolute and
scaled `L_z`/`carter_constant` drifts; the denominators use at least 1,
and the Carter scale also considers `E^2`, avoiding division by a
near-zero invariant. `E` drift is measured but should not be viewed as
independent evidence because `p_t` is deliberately fixed. Accepted
steps and SciPy RHS evaluations are reported; rejected-step count is
`None` because DOP853's public interface does not expose that count.

**Historical v0.1D chart limit, resolved in v0.1E:** the tested inclined Kerr ray approaches a
different inner-horizon branch with rapidly growing coordinate momenta in this
single ingoing chart. The diagnostic script includes such a case: it
returns `NUMERICAL_ERROR` with failing conservation diagnostics. It is
not a verified continuation into that region. General signed-r and
inner-horizon chart extensions needed their own validated stage.

## v0.1E: signed-r sheets, inner-region chart switching

The geometry owns the signed radial root, the finite disk limit,
`Sigma=r²+a²cos²(theta)`, analytic inverse-metric derivatives on both
sheets, and two stationary Cartesian KS patches. The integrator asks
the geometry for a short continuous disk extension before crossing,
then adopts the new fixed signed sheet. It selects the outgoing patch
after an inner radial turn (before an outward inner-horizon crossing),
or after a negative-to-positive disk crossing that is headed toward the
inner horizon. If an outward patch later turns inward outside the outer
horizon, it switches back to the ingoing patch. The coordinate and
covector handoff uses the **full Jacobian** of

    dt_out = dt_in - 4r/Delta dr
    dphi_out = dphi_in - 2a/Delta dr
    Delta = r² - 2r + a².

The radial coordinate and affine parameter are continuous. Each handoff
anchors the integration constants for time and azimuth at its finite
switch radius. The resulting recorded `(t,x,y,z,p_t,p_x,p_y,p_z)` states
are **piecewise chart coordinates**. Consult the same-length
`sample_charts`, `sample_branches` and `sample_regions` arrays before
comparing Cartesian components on opposite sides of a chart handoff.
`region_transitions` includes `DISK_CROSSING`, horizon events and
`CHART_HANDOFF`; horizon events also retain the original
`horizon_crossings` interface. Region names identify radial bands and
sheet signs, not a complete Penrose block index.

For regular disk crossings `Sigma` stays strictly positive; at a ring
approach the finite `Sigma=epsilon_sigma` threshold still applies.
Escape is the **outward** crossing of `abs(r)=escape_radius`, including
the negative sheet. A `DIAGNOSTIC_FAILURE` status explicitly stops a
trajectory once any of scaled null residual, scaled `L_z` drift or
scaled `carter_constant` drift exceeds `1e-8`. This is a reliability
guard, not a physical event. A handoff that cannot be performed safely
away from the horizon returns `CHART_BOUNDARY`; the horizon itself is
always nonterminal when represented in its supported patch.

`chart_boundary_radius` remains in settings for geometry interface
compatibility; signed Kerr uses it only if an implementation supplies a
chart boundary function. The exact ring and Schwarzschild point remain
distinct. DOP853, the seven evolved variables and the pointwise
Hamiltonian API remain unchanged. All chart/metric formulas are inside
`kerr_schild.py`; `hamiltonian.py` uses only `Geometry`.

The continuation covers verified regular disk passages and selected
inner/outer horizon branches, not the complete maximal Kerr extension.
Near the ring, turning rays can be sensitive to tolerances; those
violating the stated diagnostic gate are returned as unreliable rather
than presented as completed trajectories. Event detection brackets sign
changes on accepted steps, so multiple unresolved roots in one step
remain a limitation. `minimum_sampled_abs_r` and
`minimum_sampled_sigma` are sampled minima. A time-slice Eulerian launch
requires `g^tt<0`; users may initialize a negative-sheet ray only at a
position meeting the existing frame preconditions. See `VALIDATION.md`
and `scripts/v0_1e_diagnostics.py` for reproducible measurements.

The choice of ingoing versus outgoing horizon-penetrating patches for
opposite radial horizon directions is described in
[Bakun et al. (2024)](https://arxiv.org/html/2409.03722v2).

## v0.1F: extremal validation and reusable headless solves

`solve_ray_for_spin(ray_definition, a_star, settings)` integrates one fresh
launch. `sweep_ray_over_spins(ray_definition, spins, settings)` returns
results in the supplied spin order; `solve_rays(ray_definitions, a_star,
settings)` solves a batch sequentially with a shared geometry for that
spin. All calls use the existing `RayDefinition` fields: fixed Cartesian
launch point, unit local direction, Eulerian symmetric-triad convention,
coordinate time and `omega_local=1`. At **each** spin the geometry,
observer, local covector, `E`, `L_z` and `carter_constant` are rebuilt.
No old momentum or conserved quantity is reused.

    from kerr_geodesics import (
        IntegrationSettings, RayDefinition, assess_trajectory,
        solve_rays, sweep_ray_over_spins,
    )
    launch = RayDefinition([4., .5, .6], [-1., 0., 0.])
    settings = IntegrationSettings(escape_radius=8., max_affine_parameter=20.)
    sweep = sweep_ray_over_spins(launch, [.9999, 1., 1.0001], settings)
    for result in sweep:
        print(result.spin, result.status, assess_trajectory(result).level)
    batch = solve_rays([launch, RayDefinition([4., 0., 0.], [1., 0., 0.])], 1., settings)

`assess_trajectory(result)` returns `COMPLETE`, `PARTIAL` or `UNRELIABLE`.
`COMPLETE` applies only to a finite outward escape boundary or finite
`Sigma` approach with successful solver and all three scaled invariants
at or below the fixed `1e-8` guard. `PARTIAL` applies to the sampled
prefix at a chart boundary or a finite integration limit; it has no
physical terminal fate. A solver error, explicit diagnostic failure,
corrupt samples, or an exceeded guard is `UNRELIABLE` and has no physical
fate. The `1e-8` threshold was adopted in v0.1E as a conservative
empirical rejection check on the **scaled** null Hamiltonian residual
and scaled `L_z`/Carter drifts; it is not a proven global error bound.
`p_t` is exactly held fixed by stationarity, so `E` drift is not an
independent solver check. A reliable classification near a capture
boundary should also be checked against tighter solver tolerances.

Exact `a=1` has one coincident horizon root, with one outer-horizon
event per actual crossing and no artificial inner event. An inclined
ray can cross that root inward, turn in the interior, change into the
outgoing KS patch and cross outward. On the sub-extremal side, geometry
can request a handoff inside the two-horizon band when the sign of
`P(r_-) = E(r_-²+a²) - a L_z` makes the incoming inner-horizon
branch incompatible with the ingoing patch. For super-extremal spins
`Delta=r²-2r+a²` is positive; a compatible ingoing/outgoing handoff
can also cover regular radial turns there. These choices remain in the
geometry layer; DOP853, canonical momentum handoff and invariant
guards are unchanged. Horizon crossings remain nonterminal, and no
horizon events are created at `a>1`.

The v0.1F diagnostic script prints five exact-extremal rays, nine-spin
headless sweep with per-spin launch invariants and events, capture versus
escape for a fixed launch, regular super-extremal disk crossings,
trajectory separation on a **common affine interval before any chart
handoff**, 1/2/5/10-ray CPU timings at two spin updates, and a
`cProfile` sample. The continuity comparison resamples *already
integrated* positions with cubic Hermite interpolation constrained by
the Hamiltonian position derivatives; it does not infer a trajectory
at an unintegrated spin. Use `--json-output path.json` to retain the
measurements. The benchmark uses the original `rtol=1e-10` integration
model; optional timing repetition counts do not change its physics.

**Historical note:** the user subsequently ran 60/60 v0.1F tests on
Windows 11 / Python 3.14.4 / NumPy 2.5.3 / SciPy 1.18.1 and
confirmed the capture→escape diagnostics. The v0.2A compiled
backend itself has not yet been verified on that Windows environment.

## v0.2A: compiled CPU callback and headless requests

`KerrSchildGeometry` owns an optional fused scalar Numba kernel for its
**existing** analytic Hamiltonian RHS. The kernel shares `r`, `Sigma`,
Kerr–Schild ell, and derivative intermediates within one evaluation;
it never caches values across DOP853 states or spins. It algebraically
contracts the same inverse metric/derivative expressions instead of
constructing temporary 4×4 and 4×4×4 NumPy arrays for each ODE
callback. The public pointwise `metric`, `inverse_metric`, analytic
derivatives and `null_geodesic_rhs` stay on the independent v0.1F
reference path. `Geometry.fast_hamiltonian_rhs` is an optional
geometry-owned extension consumed by the integrator. Subclasses that
override geometry methods automatically use the reference path.
Neither DOP853, event semantics, tetrads, conserved quantities nor
the `1e-8` invariant guard were relaxed or replaced.

Numba `0.67.x` is an **optional performance extra**. Without it the
default `backend="auto"` uses the unmodified reference evaluation;
`backend="reference"` always forces it. With Numba installed, `auto`
uses the compiled callback for the concrete Kerr geometry, while
`backend="compiled"` requires it. At application startup call
`warm_compute_backend()` before accepting any slider-like requests;
its returned seconds measure JIT load/compile, or `None` when the
extra is not installed. `LatestWinsScheduler.start()` warms itself.

`RayBatchExecutor(workers=4, backend="compiled")` is a persistent
**process** pool, explicitly started before warmed timing or requests.
The ray-level thread pool was measured and discarded because it made
these batches slower. Each process pre-warms its own kernel and
reconstructs its own RayDefinition and geometry; the returned tuple
preserves input ray order. Use it inside `if __name__ == "__main__":`
on Windows so the spawn-based worker startup is safe. Default
`solve_rays` and `solve_ray_for_spin` remain deterministic sequential
paths; selected physical outcomes must still pass `assess_trajectory`.

    from kerr_geodesics import (
        IntegrationSettings, RayBatchExecutor, RayDefinition,
        warm_compute_backend,
    )
    if __name__ == "__main__":
        print("JIT startup seconds:", warm_compute_backend())
        rays = [RayDefinition([4., 0., 0.], [1., 0., 0.])]
        with RayBatchExecutor(workers=4, backend="compiled") as pool:
            results = pool.solve_rays(rays, 1.0, IntegrationSettings(escape_radius=8.))

`LatestWinsScheduler` is a separate, renderer-free coordinator:
`submit(spin, rays, settings)` deep-copies each saved RayDefinition,
assigns a generation and deterministic `ray-0000`-style ID unless an
explicit `RayEntry(ray_id, definition)` is supplied, and keeps at most
one pending request. New submissions discard stale queued requests,
signal a cooperative cancellation check in a running DOP853 RHS, and
make prior output ineligible for publication. `wait_for(request_id)`
reports `DONE`, `SUPERSEDED`, `CANCELLED` or `FAILED` at the **compute**
level. Only the newest `DONE` snapshot is returned by `latest()`;
its arrays are read-only. These compute statuses are not physics
termination statuses. A `DONE` batch may still contain a ray with
`DIAGNOSTIC_FAILURE`, whose physical trust must be inspected separately.

The coordinator currently performs cancellable sequential solves;
`RayBatchExecutor` separately supplies fast synchronous parallel
batches. The two are not yet a shared cancellable process queue.
Discarding a fully running process-pool batch remains a later
composition concern, not an unimplemented claim of cancellation.
Progressive/partial trajectory publication was deferred to keep the
control contract simple. That v0.2A compute layer has no UI, CUDA path, GPU solver,
preview-accuracy shortcut or Kerr–Newman charge physics.

The reproducible benchmark script prints environment, JIT warmup,
reference/compiled sequential results, process-worker timings,
profiling, trajectory equivalence, scheduler cancellation and the
test count. Exact measured Work results, remaining bottlenecks,
optional dependency compatibility and the outstanding **v0.2A-only**
Windows check are documented in `VALIDATION.md`.

## v0.3A static 3D viewer

For a desktop with OpenGL 3.3 core profile, install the optional extras:

    py -m pip install -e ".[viewer-performance]"
    py scripts/v0_3a_viewer.py --spin 0.95

Use `--spin 0`, `0.7`, `1`, `1.05` or `1.2` for the other fixed diagnostic
scenes. `python -m kerr_viewer --spin 0.95` is equivalent. This is **not** a
visual spin slider: restarting with another spin solves the same three saved
ray definitions afresh on a background worker. The UI draws the immutable
results when that worker finishes and never runs the solver inside paintGL.
The viewer uses the compiled physics backend when Numba is installed; base
`.[viewer]` uses reference physics automatically.

Controls: right mouse drag turns view; left mouse drag orbits the origin;
WASD flies; Q/E moves along global z; mouse wheel adjusts movement speed;
Escape closes. This is a Euclidean inspection camera, not a physical tetrad.
The side panel reports spin/regime/horizon radii, backend availability,
FPS, and each ray's measured fate/reliability. Default rays at a=0.95:
exterior outward escape; inclined ray crossing Kerr horizons with a chart
handoff and escape; and an off-axis regular-disk crossing traversing
positive and negative r before escaping.

The horizon is an oblate r=r+ mesh rendered with alpha blending without
depth writes. Pixel-width camera-facing colored ribbons draw **after** the
horizon so interior segments remain visible. The red point/ring and thin
+z/-z axis are purely visual markers. Outgoing-chart x/y values are mapped
by the physics geometry into the initial ingoing Cartesian chart before
forming each polyline. The result's float64 states are unchanged; only
GPU vertices use float32. No ray launch, live recompute, CUDA or Q_bh physics
is present.

### Official Windows viewer target: Python 3.13

The user's current local environment is Windows 11 / Python 3.13.x with
NumPy 2.5.x, SciPy 1.18.x and Numba 0.67.x. Binary-only Windows x86-64
wheel resolution for CPython 3.13 succeeded for ModernGL 5.12.0,
glcontext 3.0.0 and PySide6 6.11.2. Python 3.14 viewer support is no
longer a checkpoint requirement; the 3.14 notes in historical v0.3A
validation below document the superseded target.

### v0.3A-r1 ray readability

The saved diagnostic ray definitions are unchanged. Fixed palette: yellow
Exterior escape, cyan Inclined horizon return, magenta Signed-r disk
passage. The side panel matches each ray with a colored swatch, physical
status, COMPLETE/PARTIAL/UNRELIABLE assessment, signed branch/chart list,
recorded events and a short terminal explanation. ESCAPED means reaching
the *finite* configured escape boundary. SINGULARITY_APPROACH stops at
the Sigma threshold; the drawn ray is never extended into the marker.

START is a filled same-color circle with a light central dot. ESCAPED is
a hollow endpoint ring; Sigma approach is a filled red endpoint;
PARTIAL is an amber square; unreliable failures have distinct red error
glyphs and dimmed polylines. A cancelled/superseded compute has no
physical endpoint marker. Four sparse screen-facing chevrons follow
recorded sample order. Small blue/purple horizon diamonds, white disk
squares and orange chart rings mark recorded events; panel labels preserve
OUTER-I/O and INNER-I/O directly from physics events. Press `M` to
hide/show start/end, arrow and event overlays without altering the ray
polylines. Renderer-only palette and glyph logic never enters RayDefinition
or the physics backend.

Install and run on the new Windows Python 3.13 target:

    py -3.13 -m pip install -e ".[viewer-performance]"
    py -3.13 -m unittest discover -s tests -v
    py -3.13 scripts/v0_3a_viewer.py --spin 0.95
