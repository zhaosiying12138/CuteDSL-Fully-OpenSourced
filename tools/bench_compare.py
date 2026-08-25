#!/usr/bin/env python3
"""bench_compare.py — dual-stack performance comparison (official vs self).

Runs the SAME unmodified upstream benchmark script in BOTH environments:
  - .venv-reference : official nvidia-cutlass-dsl compiler
  - .venv-self      : this repo's fully open stack
Parses the script's own timing/throughput lines and emits
artifacts/perf/comparison.{json,md}.

Usage:
  .venv-self/bin/python tools/bench_compare.py [--filter elementwise]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BENCHES = {
    "elementwise-1024x1024": {
        "script": "third_party/cutlass/examples/python/CuTeDSL/cute/ampere/kernel/elementwise/elementwise_add.py",
        "argv": ["--M", "1024", "--N", "1024", "--benchmark",
                 "--warmup_iterations", "5", "--iterations", "300"],
        "metric": "throughput_gb_s",
    },
    "elementwise-2048x2048": {
        "script": "third_party/cutlass/examples/python/CuTeDSL/cute/ampere/kernel/elementwise/elementwise_add.py",
        "argv": ["--M", "2048", "--N", "2048", "--benchmark",
                 "--warmup_iterations", "5", "--iterations", "300"],
        "metric": "throughput_gb_s",
    },
    "elementwise-8192x8192": {
        "script": "third_party/cutlass/examples/python/CuTeDSL/cute/ampere/kernel/elementwise/elementwise_add.py",
        "argv": ["--M", "8192", "--N", "8192", "--benchmark",
                 "--warmup_iterations", "5", "--iterations", "300"],
        "metric": "throughput_gb_s",
    },
}

ENVS = {
    "official": ROOT / ".venv-reference/bin/python",
    "self": ROOT / ".venv-self/bin/python",
}

SELF_PYTHONPATH = f"{ROOT / 'python'}:{ROOT / 'python/cutlass_compat'}"

_TIME_RE = re.compile(r"Kernel execution time:\s*([\d.]+)\s*ms")
_THR_RE = re.compile(r"Achieved memory throughput:\s*([\d.]+)\s*GB/s")


def run_once(env: str, bench: dict, timeout: int = 1800) -> dict:
    script = ROOT / bench["script"]
    cmd = [str(ENVS[env]), str(script), *bench["argv"]]
    env_vars = None
    if env == "self":
        env_vars = {"PYTHONPATH": SELF_PYTHONPATH, "PATH": "/usr/bin:/bin"}
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env_vars,
                          capture_output=True, text=True, timeout=timeout)
    dt = time.monotonic() - t0
    out = proc.stdout + proc.stderr
    if proc.returncode != 0 or "PASS" not in out:
        return {"status": "FAIL", "returncode": proc.returncode,
                "tail": out[-500:], "wall_s": round(dt, 1)}
    ms = [float(x) for x in _TIME_RE.findall(out)]
    gbs = [float(x) for x in _THR_RE.findall(out)]
    return {
        "status": "PASS",
        "exec_ms": ms[-1] if ms else None,
        "throughput_gb_s": gbs[-1] if gbs else None,
        "verified": "Results verified successfully" in out,
        "wall_s": round(dt, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default=None)
    ap.add_argument("--repeats", type=int, default=3,
                    help="repetitions per (bench, env); median reported")
    args = ap.parse_args()

    benches = {k: v for k, v in BENCHES.items()
               if not args.filter or args.filter in k}
    results = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "repeats": args.repeats, "cases": {}}

    for name, bench in benches.items():
        results["cases"][name] = {}
        for env in ("official", "self"):
            runs = []
            for _ in range(args.repeats):
                r = run_once(env, bench)
                runs.append(r)
                print(f"[{name}/{env}] {r['status']} "
                      f"{r.get('throughput_gb_s', '-')} GB/s", flush=True)
                if r["status"] != "PASS":
                    break
            good = [r for r in runs if r["status"] == "PASS"]
            agg = dict(runs[-1])
            if good and good[0].get("throughput_gb_s") is not None:
                for key in ("exec_ms", "throughput_gb_s"):
                    vals = [r[key] for r in good if r[key] is not None]
                    if vals:
                        agg[key + "_median"] = round(statistics.median(vals), 3)
            results["cases"][name][env] = agg

    # comparison ratios
    out_md = ["# Dual-stack performance comparison (RTX 5090 Laptop)",
              "",
              f"Generated: {results['generated_at']} — median of {args.repeats} runs.",
              "",
              "| Benchmark | Official (ms) | Self (ms) | Official GB/s | Self GB/s | Self/Official |",
              "|---|---|---|---|---|---|"]
    for name, envs in results["cases"].items():
        off = envs.get("official", {})
        slf = envs.get("self", {})
        if off.get("status") == "PASS" and slf.get("status") == "PASS":
            ratio = (slf.get("throughput_gb_s_median") or 0) / \
                    (off.get("throughput_gb_s_median") or 1)
            out_md.append(
                f"| {name} | {off.get('exec_ms_median')} | {slf.get('exec_ms_median')} "
                f"| {off.get('throughput_gb_s_median')} | {slf.get('throughput_gb_s_median')} "
                f"| {ratio:.1%} |")
        else:
            out_md.append(f"| {name} | {off.get('status')} | {slf.get('status')} | - | - | - |")

    dest = ROOT / "artifacts/perf"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "comparison.json").write_text(json.dumps(results, indent=2))
    (dest / "comparison.md").write_text("\n".join(out_md) + "\n")
    print(f"\nwrote {dest}/comparison.json and comparison.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
