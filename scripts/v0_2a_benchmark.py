"""v0.2A headless CPU baseline, optimized backend and profiling measurements."""

import argparse
import cProfile
import json
from math import sqrt
from pathlib import Path
import platform
import pstats
import re
import subprocess
import sys
from threading import Event
from time import perf_counter

import numpy as np
import scipy

from kerr_geodesics import (
    ComputeStatus, IntegrationSettings, LatestWinsScheduler, RayBatchExecutor,
    RayDefinition, RayEntry, assess_trajectory, solve_ray_for_spin,
    solve_rays, warm_compute_backend,
)
from kerr_geodesics.integration import ComputationCancelled


SETTINGS = IntegrationSettings(escape_radius=8., max_affine_parameter=20.)


def definition(position, direction):
    normalized = np.asarray(direction, dtype=float)
    normalized /= np.linalg.norm(normalized)
    return RayDefinition(position, normalized)


def representative_rays():
    """Exactly the ordered ten-ray bank used in the v0.1F diagnostic."""
    return (
        definition([4., .1, .5], [1., 0., 0.]),
        definition([4., 0., 0.], [-1., 0., 0.]),
        definition([4., .5, .6], [-1., 0., 0.]),
        definition([.15, .05, 3.], [0., 0., -1.]),
        definition([.8, .1, 3.], [0., 0., -1.]),
        definition([3., .2, 1.], [1., 0., 0.]),
        definition([4., 0., 0.], [-sqrt(.75), .5, 0.]),
        definition([0., 0., 3.], [0., 0., -1.]),
        definition([1.3, .1, 3.], [0., 0., -1.]),
        definition([4., 1., .6], [.7, .2, .35]),
    )


def timing(solve, ray_bank, repeats=7, label="sequential"):
    rows = []
    for spin in (.9999, 1.2):
        solve(ray_bank[:2], spin, SETTINGS)
        for count in (1, 2, 5, 10):
            measurements = []
            for _ in range(repeats):
                start = perf_counter()
                results = solve(ray_bank[:count], spin, SETTINGS)
                measurements.append(perf_counter()-start)
                assert len(results) == count
            row = {"backend": label, "spin": spin, "rays": count,
                   "repeat": repeats, "median_s": float(np.median(measurements)),
                   "p95_s": float(np.percentile(measurements, 95)),
                   "total_s": float(sum(measurements)),
                   "median_s_per_ray": float(np.median(measurements)/count),
                   "samples_s": measurements,
                   "statuses": [item.status.value for item in results]}
            rows.append(row)
            print(f"{label:15s} a={spin:.4f} n={count:2d} "
                  f"median={row['median_s']:.4f}s p95={row['p95_s']:.4f}s "
                  f"total={row['total_s']:.4f}s", flush=True)
    return rows


def profile(solve, ray_bank):
    profiler = cProfile.Profile()
    profiler.enable()
    outputs = solve(ray_bank[:5], .9999, SETTINGS)
    profiler.disable()
    measured = pstats.Stats(profiler).stats
    keys = (
        "radial_coordinate", "_radial", "metric", "inverse_metric",
        "inverse_metric_derivatives", "null_geodesic_rhs", "kerr_schild_fields",
        "domain", "horizon_chart_handoff", "chart_transition_surface",
        "rhs", "measure", "record", "radial_speed", "integrate_ray",
        "_step_impl", "rk_step", "eulerian_frame",
    )
    selected = []
    for (filename, lineno, function), (_, calls, own, cumulative, _) in measured.items():
        if any(key in function for key in keys):
            selected.append({"file": Path(filename).name, "line": lineno,
                             "function": function, "calls": calls,
                             "self_s": own, "cumulative_s": cumulative})
    selected.sort(key=lambda entry: entry["cumulative_s"], reverse=True)
    print("PROFILE (cProfile changes absolute timings):", flush=True)
    for entry in selected[:38]:
        print(f"  {entry['file']}:{entry['line']} {entry['function']} "
              f"n={entry['calls']} self={entry['self_s']:.4f}s "
              f"cum={entry['cumulative_s']:.4f}s", flush=True)
    return {"five_ray_statuses": [result.status.value for result in outputs],
            "functions": selected}


def difficult_single_rays(bank, repeats):
    """Prevent the easy-first-ray batch row from implying all rays cost 10 ms."""
    cases = (("easy exterior", 0, .9999),
             ("inclined inner return", 2, .9999),
             ("capture inner horizon", 6, .95),
             ("super-extremal double disk", 8, 1.2))
    rows = []
    for label, index, spin in cases:
        samples = []
        for _ in range(repeats):
            start = perf_counter()
            result = solve_ray_for_spin(bank[index], spin, SETTINGS, backend="compiled")
            samples.append(perf_counter()-start)
        row = {"ray": label, "spin": spin, "status": result.status.value,
               "reliability": assess_trajectory(result).level.value,
               "median_s": float(np.median(samples)),
               "p95_s": float(np.percentile(samples, 95)),
               "samples_s": samples}
        rows.append(row)
        print("single difficulty", label, "a", spin,
              f"median={row['median_s']:.4f}s p95={row['p95_s']:.4f}s", flush=True)
    return rows


def probe_parallel(bank, repeats):
    records = []
    for workers in (2, 4, 6):
        startup = perf_counter()
        with RayBatchExecutor(workers, backend="compiled") as pool:
            startup_s = perf_counter()-startup
            print("workers ready", "process", workers, f"startup={startup_s:.3f}s", flush=True)
            for spin in (.9999, 1.2):
                for count in (5, 10):
                    samples = []
                    for _ in range(repeats):
                        start = perf_counter()
                        results = pool.solve_rays(bank[:count], spin, SETTINGS)
                        samples.append(perf_counter()-start)
                    row = {"mode": "process", "workers": workers,
                           "startup_s": startup_s, "spin": spin, "rays": count,
                           "median_s": float(np.median(samples)),
                           "p95_s": float(np.percentile(samples, 95)),
                           "samples_s": samples,
                           "statuses": [r.status.value for r in results]}
                    records.append(row)
                    print(f"probe process workers={workers} a={spin:.4f} "
                          f"n={count} median={row['median_s']:.4f}s "
                          f"p95={row['p95_s']:.4f}s", flush=True)
    return records


def equivalence(bank, pool):
    """Full reference/compiled/process trajectories and event contracts."""
    rows = []
    for spin in (0., .7, .95, .9999, 1., 1.05, 1.2):
        subset = (bank if spin in (.9999, 1.2)
                  else bank[:7] if spin in (.95, 1.05) else bank[:5])
        reference = solve_rays(subset, spin, SETTINGS, backend="reference")
        compiled = solve_rays(subset, spin, SETTINGS, backend="compiled")
        parallel = pool.solve_rays(subset, spin, SETTINGS)
        for index, (one, two, three) in enumerate(zip(reference, compiled, parallel)):
            horizon = lambda result: [(e.kind, e.direction, e.branch) for e in result.horizon_crossings]
            transitions = lambda result: [(event.kind, event.from_region, event.to_region,
                                           event.from_chart, event.to_chart)
                                          for event in result.region_transitions]
            affine_delta = max(abs(one.affine_parameter_end-two.affine_parameter_end),
                               abs(one.affine_parameter_end-three.affine_parameter_end))
            final_position_delta = max(
                float(np.max(abs(one.states[-1, 1:4]-two.states[-1, 1:4]))),
                float(np.max(abs(one.states[-1, 1:4]-three.states[-1, 1:4]))),
            )
            matched = (one.status == two.status == three.status
                       and assess_trajectory(one).level == assess_trajectory(two).level
                       == assess_trajectory(three).level
                       and horizon(one) == horizon(two) == horizon(three)
                       and transitions(one) == transitions(two) == transitions(three)
                       and one.sample_branches[-1] == two.sample_branches[-1]
                       == three.sample_branches[-1]
                       and affine_delta < 2e-6
                       and final_position_delta < 2e-5)
            rows.append({"spin": spin, "ray_id": f"ray-{index:04d}",
                         "reference_status": one.status.value,
                         "compiled_status": two.status.value,
                         "parallel_status": three.status.value,
                         "reference_trust": assess_trajectory(one).level.value,
                         "compiled_trust": assess_trajectory(two).level.value,
                         "parallel_trust": assess_trajectory(three).level.value,
                         "max_affine_difference": affine_delta,
                         "max_terminal_position_difference": final_position_delta,
                         "matches": matched})
    result = {"cases": len(rows), "all_matched": all(row["matches"] for row in rows),
              "max_affine_delta": max(row["max_affine_difference"] for row in rows),
              "max_terminal_position_delta": max(
                  row["max_terminal_position_difference"] for row in rows),
              "rows": rows}
    print("physics equivalence", result["cases"], "cases; matched",
          result["all_matched"], "max Δλ", result["max_affine_delta"], flush=True)
    if not result["all_matched"]:
        raise AssertionError("compiled/process physics-equivalence failure")
    return result


def scheduler_demo(bank):
    started = Event()
    def controlled_solver(ray, spin, settings, *, backend, cancel_requested):
        if spin == .95:
            started.set()
            while not cancel_requested():
                Event().wait(.001)
            raise ComputationCancelled("superseded")
        return solve_ray_for_spin(ray, spin, settings, backend=backend,
                                  cancel_requested=cancel_requested)
    with LatestWinsScheduler(backend="compiled", solver=controlled_solver) as coordinator:
        first = coordinator.submit(.95, (RayEntry("main", bank[0]),), SETTINGS)
        if not started.wait(3.):
            raise TimeoutError("scheduler did not start first generation")
        second = coordinator.submit(.96, (bank[0],), SETTINGS)
        third = coordinator.submit(.97, (RayEntry("main", bank[0]),), SETTINGS)
        snapshots = [coordinator.wait_for(item.request_id, 5.)
                     for item in (first, second, third)]
        result = {"request_ids": [item.request_id for item in (first,second,third)],
                  "status": [item.status.value for item in snapshots],
                  "latest_status": coordinator.latest().status.value,
                  "latest_ray_id": coordinator.latest().rays[0].ray_id,
                  "stale_published": any(item.publishable for item in snapshots[:-1])}
    if result["status"] != ["SUPERSEDED", "SUPERSEDED", "DONE"] or result["stale_published"]:
        raise AssertionError("latest-request-wins scheduler contract failed")
    print("latest wins", result, flush=True)
    return result


def regression_suite():
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = completed.stdout+completed.stderr
    count = re.search(r"Ran (\d+) tests", output)
    result = {"test_count": int(count.group(1)) if count else None,
              "passed": completed.returncode == 0,
              "command": command, "output": output[-1500:]}
    print("regression", result["test_count"], "passed", result["passed"], flush=True)
    if not result["passed"]:
        raise AssertionError("regression suite failed: " + output[-2000:])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "optimized", "parallel-probe"), default="baseline")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json-output", type=Path)
    options = parser.parse_args()
    if options.repeat < 2:
        parser.error("--repeat must be >=2")
    if options.workers < 1:
        parser.error("--workers must be >=1")
    print("environment", platform.platform(), "Python", platform.python_version(),
          "NumPy", np.__version__, "SciPy", scipy.__version__, flush=True)
    print("phase", options.phase, "rtol", SETTINGS.rtol, "ray order: v0.1F",
          flush=True)
    bank = representative_rays()
    warmup = None
    if options.phase != "baseline":
        warmup = warm_compute_backend()
        if warmup is None:
            raise RuntimeError("numba optional dependency is missing; install the performance extra")
        print("JIT warmup seconds", warmup, flush=True)
    selected = "compiled" if options.phase != "baseline" else "reference"
    def solve(bank, spin, settings):
        return solve_rays(bank, spin, settings, backend=selected)
    report = {"phase": options.phase, "platform": platform.platform(),
              "python": platform.python_version(), "numpy": np.__version__,
              "scipy": scipy.__version__, "warmup_s": warmup}
    if options.phase == "parallel-probe":
        report["worker_probe"] = probe_parallel(bank, options.repeat)
    else:
        report["benchmark"] = timing(solve, bank, options.repeat, label=selected)
        report["profile"] = profile(solve, bank)
        if options.phase == "optimized":
            report["single_ray_difficulty"] = difficult_single_rays(bank, options.repeat)
            startup = perf_counter()
            with RayBatchExecutor(options.workers, backend="compiled") as pool:
                report["worker_count"] = options.workers
                report["worker_startup_s"] = perf_counter()-startup
                print("process workers ready", options.workers,
                      "startup", report["worker_startup_s"], flush=True)
                report["parallel_benchmark"] = timing(
                    pool.solve_rays, bank, options.repeat,
                    label=f"process-{options.workers}",
                )
                report["physics_equivalence"] = equivalence(bank, pool)
            report["latest_request_wins"] = scheduler_demo(bank)
            report["regression"] = regression_suite()
    if options.json_output:
        options.json_output.parent.mkdir(parents=True, exist_ok=True)
        options.json_output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print("report", options.json_output, flush=True)


if __name__ == "__main__":
    main()
