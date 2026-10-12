#!/usr/bin/env python3
"""Aggregate run results and produce the final comparison report."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np


def load_run(out_glob):
    runs = []
    for f in sorted(glob.glob(f"{out_glob}/metrics_*.json")):
        with open(f) as fh:
            d = json.load(fh)
        runs.append(d)
    return runs


def paired_diff(a, b, key="test_acc"):
    """a, b are lists of run dicts matched by seed. Returns per-seed diffs."""
    am = {r["seed"]: r for r in a}
    bm = {r["seed"]: r for r in b}
    seeds = sorted(set(am) & set(bm))
    return [(s, am[s][key] - bm[s][key]) for s in seeds]


def summarise(runs):
    by = defaultdict(list)
    for r in runs:
        by[r["condition"]].append(r)

    out = {}
    for cond, rs in by.items():
        def stats(key):
            v = [r[key] for r in rs if isinstance(r.get(key), (int, float)) and r[key] == r[key]]
            if not v:
                return {"mean": None, "std": None, "n": 0, "values": []}
            return {
                "mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "n": len(v),
                "values": [float(x) for x in v],
            }
        out[cond] = {
            "params": rs[0]["params"],
            "test_acc": stats("test_acc"),
            "test_f1": stats("test_f1"),
            "test_nll": stats("test_nll"),
            "val_acc": stats("val_acc"),
            "retention": stats("retention"),
            "forgetting": stats("forgetting"),
            "grad_evals": stats("grad_evals"),
            "probe_grad_evals": stats("probe_grad_evals"),
            "examples_seen": stats("examples_seen"),
            "wall_s": stats("wall_s"),
            "peak_mem_mb": stats("peak_mem_mb"),
        }
    return out


def main():
    out_glob = sys.argv[1]
    runs = load_run(out_glob)
    print(f"loaded {len(runs)} runs from {out_glob}")
    s = summarise(runs)

    print("\n=== PER-CONDITION (mean +/- std over seeds) ===")
    hdr = f"{'condition':14s} {'params':>10s} {'test_acc':>16s} {'test_f1':>16s} {'val_acc':>16s} {'retention':>16s} {'grads':>7s} {'wall_s':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for cond in ["adamw", "hlc_fixed", "hlc_adaptive", "sihlc"]:
        if cond not in s:
            continue
        c = s[cond]
        def fmt(d):
            if d["mean"] is None:
                return "n/a"
            return f"{d['mean']:.4f}+/-{d['std']:.4f}"
        print(f"{cond:14s} {c['params']:>10,d} {fmt(c['test_acc']):>16s} "
              f"{fmt(c['test_f1']):>16s} {fmt(c['val_acc']):>16s} "
              f"{fmt(c['retention']):>16s} {c['grad_evals']['mean']:>7.0f} {c['wall_s']['mean']:>7.1f}")

    print("\n=== PAIRED DIFFERENCES vs adamw (per-seed test_acc delta) ===")
    by = defaultdict(list)
    for r in runs:
        by[r["condition"]].append(r)
    if "adamw" in by:
        for cond in ["hlc_fixed", "hlc_adaptive", "sihlc"]:
            if cond not in by:
                continue
            diffs = paired_diff(by[cond], by["adamw"])
            vals = [d for _, d in diffs]
            print(f"\n{cond} - adamw:")
            for seed, d in diffs:
                print(f"   seed {seed}: {d:+.4f}")
            if vals:
                print(f"   mean {np.mean(vals):+.4f}  std {np.std(vals, ddof=1) if len(vals)>1 else 0:.4f}  n={len(vals)}")

    print("\n=== SI-HLC generational selection log ===")
    for r in runs:
        if r["condition"] != "sihlc":
            continue
        print(f"\nseed {r['seed']}:")
        for g in r.get("generations", []):
            cands = ", ".join(f"c{c['candidate']}={c['val_acc']:.4f}" for c in g["candidates"])
            print(f"  gen{g['generation']} incumbent={g['incumbent_val_acc']:.4f} | {cands} "
                  f"| accepted={g['accepted']} selected={g['selected']}")

    with open(f"{out_glob}/report.json", "w") as fh:
        json.dump(s, fh, indent=2)
    print(f"\nwrote {out_glob}/report.json")


if __name__ == "__main__":
    main()
