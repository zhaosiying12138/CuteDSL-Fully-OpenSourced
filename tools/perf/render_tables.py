#!/usr/bin/env python3
"""render_tables.py — merge dual-stack perf JSONs into tables + headline mean.

Reads artifacts/perf/<family>_{self,official}.json (written by the
tools/perf/bench_* scripts), and emits:

  * artifacts/perf/summary.md      — all families, per-shape tables,
                                     headline arithmetic mean
  * artifacts/perf/summary.json    — machine-readable version
  * artifacts/perf/dense_gemm_verbatim.json / blockscaled_verbatim.json
    (legacy combined schema, refreshed from the new captures)

Headline convention (fixed, documented in the blog/README):
  per-shape pct = official_median_us / self_median_us * 100
  family pct    = arithmetic mean over that family's PASS shapes
  headline      = arithmetic mean over the six families that have an
                  official-CuTeDSL baseline (elementwise, dense_gemm,
                  blockscaled, rmsnorm_fp4quant, add_rmsnorm_fp4quant,
                  b12x MoE). MLA is reported separately (no official CuTeDSL
                  implementation exists on sm_120a); persistent/pipeline GEMM
                  are development milestones, not operator comparisons.

Usage:
  .venv-self/bin/python tools/perf/render_tables.py [--mla-json PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PERF = ROOT / "artifacts/perf"

FAMILIES = [
    ("elementwise", "elementwise_add (FP32, Ampere demo)"),
    ("dense_gemm", "dense_gemm (FP16, tile 64x64x64)"),
    ("blockscaled", "blockscaled GEMM (NVFP4 coop, tile 128x128x128)"),
    ("norm_rmsnorm_fp4quant", "flashinfer rmsnorm_fp4quant (FP16->NVFP4)"),
    ("norm_add_rmsnorm_fp4quant", "flashinfer add_rmsnorm_fp4quant"),
    ("b12x_moe", "flashinfer b12x fused MoE (W4A16 NVFP4)"),
]

LEGACY_OUT = {
    "dense_gemm": "dense_gemm_verbatim.json",
    "blockscaled": "blockscaled_verbatim.json",
}


def load(family: str, stack: str):
    path = PERF / f"{family}_{stack}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    cases = [c for c in data.get("cases", [])
             if c.get("status", "PASS") == "PASS"
             and c.get("median_us") is not None]
    return {"data": data, "cases": cases,
            "gpu_meta": data.get("gpu_meta", {})}


def shape_key(case) -> str:
    s = case.get("shape")
    if isinstance(s, list):
        return "x".join(str(v) for v in s)
    if isinstance(s, dict):
        return " ".join(f"{k}={v}" for k, v in s.items())
    return str(s)


def metric(case, family: str) -> str:
    if "tflops" in case:
        return f"{case['tflops']} TF/s"
    if "gb_s" in case:
        return f"{case['gb_s']} GB/s"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mla-json",
                    default=str(Path.home() /
                                "sm120-cutedsl-flashmla/artifacts/perf/"
                                "mla_decode_self.json"))
    args = ap.parse_args()

    summary = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "families": {}, "headline": None}
    md = ["# Performance summary — self stack vs official CuTeDSL "
          "(RTX 5090 Laptop, sm_120a)",
          "",
          f"Generated: {summary['generated_at']}",
          ""]
    family_pcts = []

    for key, title in FAMILIES:
        off = load(key, "official")
        slf = load(key, "self")
        md.append(f"## {title}")
        md.append("")
        fam = {"title": title, "cases": [], "family_pct": None}
        if not off or not slf:
            md.append(f"*incomplete — missing {'official' if not off else 'self'} capture*\n")
            summary["families"][key] = fam
            continue

        off_by = {shape_key(c): c for c in off["cases"]}
        rows = []
        pcts = []
        for c in slf["cases"]:
            k = shape_key(c)
            o = off_by.get(k)
            if o is None:
                continue
            pct = round(o["median_us"] / c["median_us"] * 100.0, 1)
            pcts.append(pct)
            rows.append((k, o, c, pct))
        fam_pct = round(sum(pcts) / len(pcts), 1) if pcts else None
        fam["family_pct"] = fam_pct
        if fam_pct is not None:
            family_pcts.append((key, title, fam_pct))

        md.append("| shape | official µs | self µs | official | self | self/official |")
        md.append("|---|---|---|---|---|---|")
        for k, o, c, pct in rows:
            md.append(f"| {k} | {o['median_us']} | {c['median_us']} | "
                      f"{metric(o, key)} | {metric(c, key)} | "
                      f"**{pct}%** |")
        if fam_pct is not None:
            md.append("")
            md.append(f"Family mean: **{fam_pct}%** of official.")
        md.append("")
        fam["cases"] = [
            {"shape": k, "official_us": o["median_us"],
             "self_us": c["median_us"], "pct": pct}
            for k, o, c, pct in rows]
        summary["families"][key] = fam

        if key in LEGACY_OUT and rows:
            legacy = {
                "captured": summary["generated_at"],
                "gpu": "RTX 5090 Laptop (sm_120a)",
                "kernel": off["data"].get("kernel"),
                "conditions": {
                    "official_gpu_meta": off["gpu_meta"],
                    "self_gpu_meta": slf["gpu_meta"]},
                "cases": [
                    {"shape": (o.get("shape") if isinstance(o.get("shape"), list)
                               else c.get("shape")),
                     "official_us": o["median_us"],
                     "self_us": c["median_us"],
                     "official_tflops": o.get("tflops"),
                     "self_tflops": c.get("tflops"),
                     "ratio": round(o["median_us"] / c["median_us"], 2)}
                    for k, o, c, pct in rows],
            }
            (PERF / LEGACY_OUT[key]).write_text(json.dumps(legacy, indent=1))
            print(f"[render] refreshed {PERF / LEGACY_OUT[key]}")

    if family_pcts:
        headline = round(sum(p for _, _, p in family_pcts) / len(family_pcts), 1)
        summary["headline"] = {
            "convention": ("arithmetic mean over the 6 official-baselined "
                           "families of (family mean of official_us/self_us "
                           "x 100)"),
            "pct": headline,
            "families": {k: p for k, _, p in family_pcts},
        }
        md.append("## Headline")
        md.append("")
        for k, t, p in family_pcts:
            md.append(f"- {t}: **{p}%**")
        md.append("")
        md.append(f"**Arithmetic mean across families: {headline}% of "
                  f"official CuTeDSL throughput.**")
        md.append("")

    mla_path = Path(args.mla_json)
    if mla_path.exists():
        mla = json.loads(mla_path.read_text())
        md.append("## FlashMLA decode (self-built sm120 core — separate "
                  "baseline)")
        md.append("")
        md.append("No official CuTeDSL MLA exists on sm_120a (SM100 "
                  "tcgen05/TMEM only); baseline is a PyTorch "
                  "(einsum/softmax) reference:")
        md.append("")
        rows = mla.get("perf") or mla.get("cases") or []
        if rows:
            md.append("| shape | self µs | torch µs | speedup |")
            md.append("|---|---|---|---|")
            for r in rows:
                shape = r.get("shape") or [r.get("B"), r.get("S"), r.get("H")]
                md.append(f"| {shape} | {r.get('self_us')} | "
                          f"{r.get('torch_ref_us', r.get('torch_us'))} | "
                          f"**{r.get('speedup')}x** |")
            md.append("")
        summary["mla"] = mla

    md.append("## Measurement conditions")
    md.append("")
    md.append("See the `gpu_meta` blocks inside the per-family JSON files "
              "(utilization / clocks / power at capture time; shared-GPU "
              "captures are labeled there). Re-run the whole suite with "
              "`tools/perf/run_all_perf.sh`.")

    (PERF / "summary.json").write_text(json.dumps(summary, indent=1))
    (PERF / "summary.md").write_text("\n".join(md) + "\n")
    print(f"[render] wrote {PERF}/summary.json and summary.md")
    if summary["headline"]:
        print(f"[render] headline: {summary['headline']['pct']}% of official")
    return 0


if __name__ == "__main__":
    sys.exit(main())
