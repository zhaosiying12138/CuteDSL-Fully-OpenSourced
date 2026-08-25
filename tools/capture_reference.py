#!/usr/bin/env python3
"""capture_reference.py — freeze official-environment baseline evidence (M0).

Runs each case from the reference manifest in the OFFICIAL environment
(.venv-reference with nvidia-cutlass-dsl installed), records versions,
source hashes, exit status and stdout/stderr digests into
artifacts/reference/results.json.

This is the ONLY tool allowed to touch the official wheel. It never inspects
proprietary implementation details; it only executes public example scripts
and records run metadata + pass/fail.

Usage:
    .venv-reference/bin/python tools/capture_reference.py \
        --manifest compat/sm120_reference.lock.yaml \
        [--case-ids id1,id2] [--timeout 1800]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_env_meta() -> dict:
    import torch

    meta = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "driver_version": torch.version.hip if torch.version.hip else os.popen(
            "/usr/lib/wsl/lib/nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null"
        ).read().strip(),
    }
    try:
        import cutlass

        meta["official_dsl_version"] = cutlass.__version__
    except Exception:
        meta["official_dsl_version"] = None
    return meta


def run_case(case: dict, env_meta: dict, timeout: int) -> dict:
    src = ROOT / case["source_path"]
    rec = {
        "id": case["id"],
        "source_path": case["source_path"],
        "source_sha256": sha256_file(src) if src.exists() else None,
        "argv": case.get("argv", []),
        **env_meta,
        "seed": case.get("seed", 0),
        "atol": case.get("atol"),
        "rtol": case.get("rtol"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "required": case.get("required", True),
    }
    if not src.exists():
        rec.update(status="NOT_CAPTURED", detail=f"missing source {src}")
        return rec

    cmd = [sys.executable, str(src)] + [str(a) for a in case.get("argv", [])]
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", str(case.get("seed", 0)))
    log_path = ROOT / "artifacts" / "reference" / "logs" / f"{case['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.monotonic()
        with open(log_path, "w") as log:
            proc = subprocess.run(
                cmd, cwd=str(ROOT), env=env, timeout=timeout,
                stdout=log, stderr=subprocess.STDOUT,
            )
        dt = time.monotonic() - t0
        out = log_path.read_text(errors="replace")
        rec.update(
            status="PASS" if proc.returncode == 0 else "FAIL",
            returncode=proc.returncode,
            wall_seconds=round(dt, 2),
            stdout_sha256=hashlib.sha256(out.encode()).hexdigest(),
            log=str(log_path.relative_to(ROOT)),
        )
        for needle in ("Traceback", "CUDA error", "RuntimeError"):
            if proc.returncode != 0 and needle in out:
                rec["detail"] = out[-2000:]
                break
    except subprocess.TimeoutExpired:
        rec.update(status="TIMEOUT", detail=f"exceeded {timeout}s")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="compat/sm120_reference.lock.yaml")
    ap.add_argument("--case-ids", default=None, help="comma-separated subset")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--output", default="artifacts/reference/results.json")
    args = ap.parse_args()

    manifest = yaml.safe_load((ROOT / args.manifest).read_text())
    cases = manifest["cases"]
    if args.case_ids:
        wanted = set(args.case_ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    env_meta = collect_env_meta()
    results = {"captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": env_meta, "cases": []}
    out_path = ROOT / args.output
    if out_path.exists():  # resume-friendly
        prev = json.loads(out_path.read_text())
        results["cases"] = [c for c in prev.get("cases", [])]

    existing = {c["id"] for c in results["cases"]}
    for case in cases:
        if case["id"] in existing:
            print(f"[skip] {case['id']} (already captured; delete to re-run)")
            continue
        print(f"[run ] {case['id']}: {case['source_path']} {' '.join(map(str, case.get('argv', [])))}",
              flush=True)
        rec = run_case(case, env_meta, args.timeout)
        print(f"[done] {case['id']}: {rec['status']} ({rec.get('wall_seconds', '?')}s)", flush=True)
        results["cases"] = [c for c in results["cases"] if c["id"] != rec["id"]] + [rec]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    n_fail = sum(1 for c in results["cases"] if c["status"] != "PASS")
    print(f"\n{len(results['cases'])} captured, {n_fail} not-PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
