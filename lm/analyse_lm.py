#!/usr/bin/env python3
"""
analyse_lm.py -- aggregate LM experiment results with paired differences.

Usage:
  python analyse_lm.py <run_dir> [--baseline adamw]
"""
import argparse
import collections
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

COND_ORDER = ["adamw", "hlc_fixed", "hlc_adaptive", "sihlc"]


def ms(vals):
    v = [x for x in vals if x == x]
    if not v:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "n": len(v), "values": [float(x) for x in v]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", default="adamw")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.run_dir)
                   if f.startswith("metrics_") and f.endswith(".json"))
    runs = []
    for f in files:
        with open(os.path.join(args.run_dir, f)) as fh:
            runs.append(json.load(fh))
    print(f"loaded {len(runs)} runs from {args.run_dir}\n")

    by = collections.defaultdict(list)
    for r in runs:
        by[r["condition"]].append(r)

    out = {}
    hdr = (f"{'condition':13s} {'params':>12s} {'test_loss':>10s} {'test_ppl':>10s} "
           f"{'val_ppl':>9s} {'ret_ppl':>9s} {'steps':>6s} {'probe':>6s} "
           f"{'lin_tokens':>11s} {'probe_tok':>10s} {'wall_s':>7s} {'peakMB':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for c in COND_ORDER:
        if c not in by:
            continue
        rs = by[c]
        g = lambda k: ms([r.get(k, float("nan")) for r in rs])
        rec = {"params": rs[0]["params"], "test_loss": g("test_loss"),
               "test_ppl": g("test_ppl"), "val_loss": g("val_loss"),
               "val_ppl": g("val_ppl"), "retention_ppl": g("retention_ppl"),
               "retention_loss": g("retention_loss"),
               "steps": g("steps"), "probe_steps": g("probe_steps"),
               "tokens_seen": g("tokens_seen"),
               "probe_tokens": g("probe_tokens"),
               "wall_s": g("wall_s"), "peak_mem_mb": g("peak_mem_mb")}
        print(f"{c:13s} {rec['params']:>12,} {rec['test_loss']['mean']:>10.4f} "
              f"{rec['test_ppl']['mean']:>10.2f} {rec['val_ppl']['mean']:>9.2f} "
              f"{rec['retention_ppl']['mean']:>9.2f} {rec['steps']['mean']:>6.0f} "
              f"{rec['probe_steps']['mean']:>6.0f} "
              f"{rec['tokens_seen']['mean']:>11,.0f} "
              f"{rec['probe_tokens']['mean']:>10,.0f} "
              f"{rec['wall_s']['mean']:>7.0f} {rec['peak_mem_mb']['mean']:>7.0f}")
        out[c] = rec

    # paired per-seed differences vs baseline
    base = {r["seed"]: r for r in by.get(args.baseline, [])}
    seeds = sorted(base)
    print(f"\nPaired per-seed test_loss differences vs {args.baseline} "
          f"(negative = better)")
    hdr2 = f"{'condition':13s} " + " ".join(f"{'seed'+str(s):>9s}" for s in seeds) \
        + f" {'mean':>9s} {'std':>8s}"
    print(hdr2)
    print("-" * len(hdr2))
    for c in COND_ORDER:
        if c not in by or c == args.baseline:
            continue
        diffs = []
        for r in sorted(by[c], key=lambda x: x["seed"]):
            b = base.get(r["seed"])
            if b is not None:
                diffs.append(r["test_loss"] - b["test_loss"])
        if len(diffs) != len(seeds):
            print(f"{c:13s} (incomplete pairing: {len(diffs)}/{len(seeds)})")
            continue
        sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        print(f"{c:13s} " + " ".join(f"{d:>+9.4f}" for d in diffs)
              + f" {np.mean(diffs):>+9.4f} {sd:>8.4f}")
        out[c]["paired_test_loss_diff"] = {
            "values": [float(d) for d in diffs],
            "mean": float(np.mean(diffs)), "std": sd}

    # generational behaviour
    flat = [g for r in by.get("sihlc", []) for g in r.get("generations", [])]
    if flat:
        acc = sum(1 for g in flat if g["accepted"])
        print(f"\nsihlc tournaments: {acc}/{len(flat)} accepted a candidate "
              f"({100*acc/len(flat):.0f}%)")
        out["sihlc"]["tournaments"] = {"n": len(flat), "accepted": acc,
                                       "accepted_frac": acc / len(flat)}

    with open(os.path.join(args.run_dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(args.run_dir, 'analysis.json')}")


if __name__ == "__main__":
    main()
