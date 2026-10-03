"""Headless v0.1F extremal/spin-sweep diagnostics and sequential CPU baseline.

Run after ``python -m pip install -e .``; no UI or rendering dependency.
All trajectories are integrated separately. Continuity interpolation is
used solely for comparing already-computed trajectories on a shared affine
interval in the same chart; it does not create new physics trajectories.
"""

import argparse
import cProfile
import json
from math import sqrt
from pathlib import Path
import platform
import pstats
from time import perf_counter

import numpy as np
import scipy
from scipy.interpolate import CubicHermiteSpline

from kerr_geodesics import (
    IntegrationSettings, KerrSchildGeometry, RayDefinition, SpacetimeParameters,
    assess_trajectory, solve_ray_for_spin, solve_rays, sweep_ray_over_spins,
)


SPINS = (.95, .99, .999, .9999, 1., 1.0001, 1.001, 1.01, 1.05)
SETTINGS = IntegrationSettings(escape_radius=8., max_affine_parameter=20.)
BENCH_SPINS = (.9999, 1.2)


def definition(position, direction):
    unit = np.asarray(direction, dtype=float)
    unit /= np.linalg.norm(unit)
    return RayDefinition(position, unit)


def summarize(result):
    conserved = result.initial.conserved
    assessment = assess_trajectory(result)
    return {
        "spin": result.spin,
        "status": result.status.value,
        "reliability": assessment.level.value,
        "physical_outcome": assessment.physical_outcome.value if assessment.physical_outcome else None,
        "initial": {"E": conserved.energy, "Lz": conserved.angular_momentum_z,
                    "carter_constant": conserved.carter_constant},
        "lambda_end": result.affine_parameter_end,
        "max_scaled_null": result.max_scaled_null_residual,
        "max_scaled_Lz_drift": result.max_scaled_Lz_drift,
        "max_scaled_carter_drift": result.max_scaled_carter_constant_drift,
        "max_abs_Lz_drift": result.max_abs_Lz_drift,
        "max_abs_carter_drift": result.max_abs_carter_constant_drift,
        "minimum_abs_r": result.minimum_sampled_abs_r,
        "minimum_sigma": result.minimum_sampled_sigma,
        "horizons": [{"kind": e.kind.value, "direction": e.direction,
                      "lambda": e.affine_parameter} for e in result.horizon_crossings],
        "transitions": [{"kind": e.kind, "lambda": e.affine_parameter,
                         "region": [e.from_region, e.to_region],
                         "chart": [e.from_chart, e.to_chart]}
                        for e in result.region_transitions],
        "charts": list(dict.fromkeys(result.sample_charts)),
        "branches": list(dict.fromkeys(int(b) for b in result.sample_branches)),
        "accepted_steps": result.accepted_steps,
        "rhs_evaluations": result.rhs_evaluations,
        "message": result.message,
    }


def display(label, result):
    record = summarize(result)
    initial = record["initial"]
    horizons = ",".join(f"{e['kind'].split('_')[0]}-{e['direction'][0]}"
                        for e in record["horizons"]) or "none"
    transitions = ",".join(f"{e['kind']}@{e['lambda']:.4f}"
                           for e in record["transitions"]) or "none"
    print(f"{label:22s} a={result.spin:8.4f} {record['status']:23s} "
          f"trust={record['reliability']:10s} λ={record['lambda_end']:10.6f} "
          f"E={initial['E']: .6g} Lz={initial['Lz']: .6g} Q={initial['carter_constant']: .6g} "
          f"null={record['max_scaled_null']:.2e} "
          f"ΔLz={record['max_scaled_Lz_drift']:.2e} "
          f"ΔQ={record['max_scaled_carter_drift']:.2e} "
          f"h={horizons} branch={record['branches']} chart={record['charts']} "
          f"events={transitions} steps/nfev={result.accepted_steps}/{result.rhs_evaluations}")
    return record


def common_affine_coordinates(result, upper=1.2):
    """Hermite resample actual integrated positions in one initial KS chart.

    Analytic position derivatives from the recorded Hamiltonian states
    constrain the interpolant. This is for measurement only; no resampled
    state is integrated or classified. Never interpolate across a handoff.
    """
    times = result.affine_parameters
    end = int(np.searchsorted(times, upper, side="left")) + 1
    if end > len(times) or len(set(result.sample_charts[:end])) != 1:
        raise ValueError("common comparison interval crosses a chart transition")
    geometry = KerrSchildGeometry(SpacetimeParameters(a_star=result.spin))
    positions = result.states[:end, 1:4]
    velocities = np.array([
        (geometry.inverse_metric(state[1:4]) @ state[4:8])[1:4]
        for state in result.states[:end]
    ])
    spline = CubicHermiteSpline(times[:end], positions, velocities)
    return spline(np.linspace(0., upper, 61))


def continuity_survey(sweep):
    positions = [common_affine_coordinates(result) for result in sweep]
    rows = []
    for index, (left, right) in enumerate(zip(positions, positions[1:])):
        first, second = sweep[index].spin, sweep[index+1].spin
        largest = float(np.max(np.linalg.norm(right-left, axis=1)))
        rows.append({"spins": [first, second], "common_lambda": [0., 1.2],
                     "max_cartesian_distance": largest,
                     "per_unit_spin": largest / (second-first)})
        print(f"a={first:.4f}→{second:.4f} shared affine λ∈[0,1.2]: "
              f"max |Δ(x,y,z)|={largest:.8g}; per Δa={largest/(second-first):.8g}")
    return rows


def benchmark(ray_bank, repetitions):
    """Sequential same-spin batch updates at full v0.1F production accuracy."""
    rows = []
    for spin in BENCH_SPINS:
        # One warmup keeps import/bytecode costs outside the measured batch.
        solve_rays(ray_bank[:2], spin, SETTINGS)
        for count in (1, 2, 5, 10):
            samples = []
            fates = None
            for _ in range(repetitions):
                start = perf_counter()
                solutions = solve_rays(ray_bank[:count], spin, SETTINGS)
                samples.append(perf_counter()-start)
                fates = [summarize(result)["reliability"] + ":" + result.status.value
                         for result in solutions]
            row = {"spin": spin, "rays": count, "repetitions": repetitions,
                   "median_s": float(np.median(samples)),
                   "p95_s": float(np.percentile(samples, 95)),
                   "total_s": float(sum(samples)),
                   "median_s_per_ray": float(np.median(samples)/count),
                   "observations_s": samples, "outcomes": fates}
            rows.append(row)
            print(f"a={spin:.4f}, {count:2d} rays: "
                  f"median={row['median_s']:.4f}s p95={row['p95_s']:.4f}s "
                  f"sum={row['total_s']:.4f}s median/ray={row['median_s_per_ray']:.4f}s "
                  f"unreliable={sum(not fate.startswith('COMPLETE') for fate in fates)}")
    return rows


def profile_baseline(ray_bank):
    """cProfile self time by layer, plus top cumulative-time functions."""
    profiler = cProfile.Profile()
    profiler.enable()
    results = solve_rays(ray_bank[:5], .9999, SETTINGS)
    profiler.disable()
    stats = pstats.Stats(profiler).stats
    buckets = {key: 0. for key in
               ("geometry/Hamiltonian", "tetrad/launch", "integrator/events",
                "SciPy solver", "NumPy", "other")}
    top = []
    for (filename, line, name), (primitive, calls, self_time, cumulative, callers) in stats.items():
        if "kerr_schild.py" in filename or "hamiltonian.py" in filename:
            bucket = "geometry/Hamiltonian"
        elif "initial_conditions.py" in filename:
            bucket = "tetrad/launch"
        elif "integration.py" in filename or "solving.py" in filename:
            bucket = "integrator/events"
        elif "scipy/integrate" in filename or "scipy\\integrate" in filename:
            bucket = "SciPy solver"
        elif "numpy" in filename:
            bucket = "NumPy"
        else:
            bucket = "other"
        buckets[bucket] += self_time
        top.append((cumulative, self_time, Path(filename).name, line, name, calls))
    total = sum(buckets.values())
    hottest = [{"file": file, "line": line, "function": name,
                "calls": calls, "self_s": own, "cumulative_s": cumulative}
               for cumulative, own, file, line, name, calls in sorted(top, reverse=True)[:14]]
    print("cProfile self time (instrumented; percentages exclude no Python/native call category):")
    for label, seconds in buckets.items():
        print(f"  {label:21s} {seconds:8.4f}s {100*seconds/total:5.1f}%")
    print("top cumulative functions:")
    for row in hottest[:10]:
        print(f"  {row['file']}:{row['line']} {row['function']} "
              f"calls={row['calls']} cumulative={row['cumulative_s']:.3f}s "
              f"self={row['self_s']:.3f}s")
    return {"spin": .9999, "rays": len(results), "instrumented_total_s": total,
            "self_time_s_by_layer": buckets, "top_cumulative": hottest,
            "results": [summarize(result)["status"] for result in results]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=7,
                        help="per (spin, ray-count) CPU timing repetitions (default 7)")
    parser.add_argument("--json-output", type=Path,
                        help="write complete measured diagnostics JSON to this path")
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error("--repeat must be at least 2")
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "cpu": platform.processor(), "cpu_count": __import__("os").cpu_count(),
                   "numpy": np.__version__, "scipy": scipy.__version__}
    print("Environment:", environment)
    report = {"environment": environment,
              "settings": {"escape_radius": SETTINGS.escape_radius,
                           "max_affine_parameter": SETTINGS.max_affine_parameter,
                           "rtol": SETTINGS.rtol, "atol": list(SETTINGS.atol),
                           "max_step": SETTINGS.max_step},
              "trust_guard": "each of scaled null, Lz, Carter <= 1e-8; solver/chart validity"}

    print("\nEXACT EXTREMAL a=1")
    extremal_cases = (
        ("outgoing", [4., .3, .6], [1., 0., 0.]),
        ("equatorial plunge", [4., 0., 0.], [-1., 0., 0.]),
        ("inclined return", [4., .5, .6], [-1., 0., 0.]),
        ("inclined off-axis", [3., .2, 1.], [-1., 0., 0.]),
        ("near-axis disk", [.15, .05, 3.], [0., 0., -1.]),
    )
    extremal = {}
    for name, position, direction in extremal_cases:
        extremal[name] = display(name, solve_ray_for_spin(
            definition(position, direction), 1., SETTINGS))
    report["extremal"] = extremal

    print("\nNINE-SPIN INCLINED SWEEP (same RayDefinition, fresh local momentum each time)")
    ray = definition([4., .5, .6], [-1., 0., 0.])
    sweep = sweep_ray_over_spins(ray, SPINS, SETTINGS)
    report["spin_sweep"] = []
    for spin, result in zip(SPINS, sweep):
        regime = KerrSchildGeometry(SpacetimeParameters(a_star=spin)).horizons().regime.value
        row = display(regime, result)
        row["horizon_regime"] = regime
        report["spin_sweep"].append(row)

    print("\nCOMMON AFFINE COORDINATE COMPARISON (resampling measured trajectories only)")
    report["continuity"] = continuity_survey(sweep)

    print("\nCAPTURE→ESCAPE AT FIXED LOCAL LAUNCH")
    capture_ray = definition([4., 0., 0.], [-sqrt(.75), .5, 0.])
    report["capture_escape"] = {}
    for spin in (.95, 1.05):
        result = solve_ray_for_spin(capture_ray, spin, SETTINGS)
        report["capture_escape"][str(spin)] = display("same equatorial ray", result)
        tight = solve_ray_for_spin(capture_ray, spin, IntegrationSettings(
            escape_radius=8., max_affine_parameter=20., rtol=1e-11))
        print(f"  tightened rtol=1e-11: fate={tight.status.value}, "
              f"|Δλend|={abs(result.affine_parameter_end-tight.affine_parameter_end):.3e}, "
              f"trust={assess_trajectory(tight).level.value}")
        report["capture_escape"][str(spin)]["refined"] = summarize(tight)
    loose = solve_ray_for_spin(capture_ray, 1.05, IntegrationSettings(
        escape_radius=8., max_affine_parameter=20., rtol=1e-9))
    print(f"  exploratory loose rtol=1e-9 at a=1.05: "
          f"{loose.status.value}, trust={assess_trajectory(loose).level.value} "
          "(this is not an accepted preview)")
    report["exploratory_loose_preview"] = summarize(loose)

    print("\nSUPER-EXTREMAL DISK AND CLOSE PASSAGES (no horizon expected)")
    super_cases = ((1.0001, .8), (1.001, .8), (1.01, .8), (1.05, 1.), (1.2, 1.3))
    report["super_extremal"] = []
    for spin, x in super_cases:
        result = solve_ray_for_spin(definition([x, .1, 3.], [0., 0., -1.]),
                                    spin, SETTINGS)
        report["super_extremal"].append(display("disk / close passage", result))
    near_axis = solve_ray_for_spin(definition([.15, .05, 3.], [0., 0., -1.]),
                                   1.2, SETTINGS)
    report["super_extremal_near_axis"] = display("near axis a=1.2", near_axis)

    print("\nSEQUENTIAL CPU BENCHMARK: full accuracy, same-spin batch updates")
    ray_bank = (
        definition([4., .1, .5], [1., 0., 0.]),
        definition([4., 0., 0.], [-1., 0., 0.]),
        definition([4., .5, .6], [-1., 0., 0.]),
        definition([.15, .05, 3.], [0., 0., -1.]),
        definition([.8, .1, 3.], [0., 0., -1.]),
        definition([3., .2, 1.], [1., 0., 0.]),
        capture_ray,
        definition([0., 0., 3.], [0., 0., -1.]),
        definition([1.3, .1, 3.], [0., 0., -1.]),
        definition([4., 1., .6], [.7, .2, .35]),
    )
    report["benchmark_ray_bank"] = [
        {"coordinate_position": ray.coordinate_position.tolist(),
         "normalized_local_direction": ray.local_direction.tolist()}
        for ray in ray_bank
    ]
    report["benchmark"] = benchmark(ray_bank, args.repeat)

    print("\nPROFILE: first five rays at a=0.9999 (cProfile overhead, not baseline timing)")
    report["profile"] = profile_baseline(ray_bank)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n",
                                    encoding="utf-8")
        print("Wrote:", args.json_output)


if __name__ == "__main__":
    main()
