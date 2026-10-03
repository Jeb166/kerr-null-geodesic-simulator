# Kerr Null Geodesic Simulator

A local 3D simulator for exploring how light travels through stationary Kerr spacetime. The focus is physically computed null geodesics, clear geometry, and numerical diagnostics.

Compare sub-extremal, extremal, and super-extremal spins, inspect horizon crossings and signed radial branches, and follow trajectories with a free desktop camera.

## Current features

- Cartesian Kerr–Schild geometry with analytic metric derivatives.
- Float64 Hamiltonian null-geodesic integration using SciPy DOP853.
- Local ray directions defined in an Eulerian observer's orthonormal frame.
- Signed-r continuation through regular disk crossings and ingoing/outgoing chart handoffs.
- Conservation diagnostics with `COMPLETE`, `PARTIAL`, and `UNRELIABLE` assessments.
- Optional Numba acceleration, process-based batch solving, and a headless latest-request-wins scheduler.
- PySide6/ModernGL viewer with horizon geometry, singularity markers, colored trajectories, propagation arrows, and event markers.

The viewer currently displays predefined rays at a spin selected on startup. Interactive ray launch and a live spin slider are planned. The compute scheduler is available in the physics package but is not yet connected to live viewer controls.

## Setup

The official desktop target is **Windows 11 with Python 3.13** and an OpenGL 3.3 capable GPU. The local physics stack uses NumPy 2.5.x, SciPy 1.18.x, and optional Numba 0.67.x.

Run these commands to clone, install, and launch:

```powershell
git clone https://github.com/Jeb166/kerr-null-geodesic-simulator.git
cd kerr-null-geodesic-simulator
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[viewer-performance]"
.\.venv\Scripts\python.exe -m kerr_viewer --spin 0.95
```

For physics-only use, install `-e .` instead. The `[performance]` extra adds Numba without GUI dependencies; `[viewer]` installs the viewer without Numba.

## Viewer usage

Set the stationary spin with `--spin`. Useful comparison values are `0`, `0.7`, `0.95`, `1`, `1.05`, and `1.2`. Each launch rebuilds the same stored local ray definitions for the selected geometry, including the observer frame, momentum, and conserved quantities.

| Control | Action |
| --- | --- |
| Right mouse drag | Look around |
| Left mouse drag | Orbit the origin |
| W / A / S / D | Move forward / left / backward / right |
| Q / E | Move down / up along the global spin axis |
| Mouse wheel | Adjust movement speed |
| M | Toggle trajectory markers and arrows |
| Escape | Close the viewer |

The side panel shows spin, horizon radii, backend status, FPS, and each ray's fate, reliability, and events. The camera is a geometry inspection tool. Trajectories are converted into a common rendering representation without smoothing or changing the physics results.

## Physics and results

The model uses geometric units **G = c = M = 1**, spin **a = a\***, and the **+z** spin axis. Only uncharged Kerr is implemented; Kerr–Newman charge is reserved for future work.

For **|a| < 1**, nonzero spin gives distinct outer and inner horizon roots. At **|a| = 1**, they coincide at r = 1. For **|a| > 1**, there are no real horizon roots. The singularity is a point at a = 0 and a ring otherwise; regular r = 0 disk crossings are distinct from the singular ring.

Horizon crossings are recorded and do not terminate integration. Read each endpoint together with its reliability assessment:

| Result | Meaning |
| --- | --- |
| `ESCAPED` | Reached the finite outward escape boundary. |
| `SINGULARITY_APPROACH` | Reached a finite Sigma threshold near the singularity. |
| `MAX_INTEGRATION_LIMIT` | Reached the integration budget. |
| `CHART_BOUNDARY` | A required coordinate continuation was unavailable. |
| `NUMERICAL_ERROR` | The solver or a numerical evaluation failed. |
| `DIAGNOSTIC_FAILURE` | A conservation or null-condition reliability guard failed. |

Only accepted `COMPLETE` results represent a resolved finite escape or singularity approach. `PARTIAL` trajectories are unfinished; `UNRELIABLE` results do not establish a physical fate. Cancellation and supersession belong to the compute layer.

## Validation

Run the regression suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover geometry, analytic derivatives, local launches, conserved quantities, horizon and disk crossings, extremal regimes, backend equivalence, scheduling, and viewer result mapping. See [VALIDATION.md](VALIDATION.md) for detailed measurements and checkpoint records, and [scripts/](scripts/) for diagnostics and benchmarks.

## Scope and limitations

This models exact stationary Kerr geometry. Supported chart continuations are not a complete atlas of the maximal Kerr extension, and the model does not include astrophysical inner-horizon instability. Escape and singularity approach use finite numerical boundaries. Rays near the ring or a separatrix can be difficult to resolve, and multiple event crossings within one integration step can require care.

The project focuses on geometric inspection. Accretion disks, background lensing, and decorative effects are outside the current scope. Viewer trajectory interpretation is still undergoing validation.

## Code layout

- [src/kerr_geodesics/](src/kerr_geodesics/): geometry, initial conditions, integration, diagnostics, solve APIs, and compute scheduling.
- [src/kerr_viewer/](src/kerr_viewer/): desktop UI, camera, and rendering of result snapshots.
- [tests/](tests/): regression checks.
- [scripts/](scripts/): diagnostics, benchmarks, and viewer entry point.

The physics package remains independent of Qt and OpenGL.

## References

- [SpECTRE Kerr–Schild analytic solution](https://spectre-code.org/classgr_1_1Solutions_1_1KerrSchild.html)
- [Matt Visser, The Kerr spacetime](https://arxiv.org/abs/0706.0622)
- [Christopher J. White, Blacklight](https://arxiv.org/abs/2203.15963)
