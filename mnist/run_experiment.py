#!/usr/bin/env python3
"""
Driver: run the SI-HLC experiment.

Usage:
  python run_experiment.py --mode smoke     # tiny budget, 1 seed, verify plumbing
  python run_experiment.py --mode screen    # screening: all 4 conditions, 1 seed
  python run_experiment.py --mode main      # main: all 4 conditions, 5 seeds

Saves: checkpoints, config, seeds, logs, metrics per condition/seed.
"""
import argparse
import copy
import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch

from hlc_core import (
    TARGET_PARAMS,
    build_model,
    count_params,
    evaluate,
    init_weights,
    load_mnist,
    split_train_val,
    to_tensor,
)
from hlc_train import HLCTrainer, Policy, RunResult

CONDITIONS = ["adamw", "hlc_fixed", "hlc_adaptive", "sihlc"]

# matched seeds across conditions
SEEDS = [0, 1, 2, 3, 4]


def prepare_data(data_dir: str, val_size: int, seed: int, device):
    xtr, ytr, xte, yte = load_mnist(data_dir)
    xtr, ytr, xv, yv = split_train_val(xtr, ytr, val_size, seed)
    data = {
        "train": to_tensor(xtr, ytr, device),
        "val": to_tensor(xv, yv, device),
        "test": to_tensor(xte, yte, device),
    }
    # retention set: the easiest 2000 training examples (BIRTH foundation)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "screen", "main"], default="smoke")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--val-size", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"runs/{args.mode}_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "smoke":
        budget, gens, seeds, conds = 120, 1, [0], ["sihlc"]
    elif args.mode == "screen":
        budget, gens, seeds, conds = 1200, 2, [0], CONDITIONS
    else:
        budget, gens, seeds, conds = 3000, 3, SEEDS, CONDITIONS

    if args.budget is not None:
        budget = args.budget
    if args.generations is not None:
        gens = args.generations
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    if args.conditions:
        conds = args.conditions.split(",")

    cfg = {
        "mode": args.mode,
        "timestamp_utc": ts,
        "budget_steps": budget,
        "generations": gens,
        "batch_size": args.batch_size,
        "val_size": args.val_size,
        "threads": args.threads,
        "seeds": seeds,
        "conditions": conds,
        "target_params": TARGET_PARAMS,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] {json.dumps(cfg, indent=2)}")

    device = torch.device("cpu")
    results: list[RunResult] = []

    for seed in seeds:
        data = prepare_data(args.data_dir, args.val_size, seed, device)
        # retention set = easiest 2000 of the training split
        xr = data["train"][0][:2000]
        yr = data["train"][1][:2000]

        for cond in conds:
            tag = f"{cond}_seed{seed}"
            print(f"\n===== {tag} =====", flush=True)
            policy = Policy()
            trainer = HLCTrainer(
                condition=cond,
                seed=seed,
                data=data,
                policy=policy,
                batch_size=args.batch_size,
                threads=args.threads,
                out_dir=out_dir,
                retention_set=(xr, yr),
                max_generations=gens,
            )
            assert trainer.params == TARGET_PARAMS, trainer.params
            res = trainer.run(total_budget=budget)

            # checkpoint: model + optimizer + config
            ckpt = {
                "condition": cond,
                "seed": seed,
                "model_state": trainer.model.state_dict(),
                "opt_state": trainer.opt.state_dict(),
                "params": trainer.params,
                "policy": trainer.policy.to_dict(),
                "result": res.__dict__,
                "config": cfg,
            }
            cpath = os.path.join(out_dir, f"ckpt_{tag}.pt")
            torch.save(ckpt, cpath)

            # json metrics
            mpath = os.path.join(out_dir, f"metrics_{tag}.json")
            with open(mpath, "w") as f:
                json.dump(res.__dict__, f, indent=2, default=str)

            results.append(res)
            print(
                f"[{tag}] test_acc={res.test_acc:.4f} test_f1={res.test_f1:.4f} "
                f"val_acc={res.val_acc:.4f} retention={res.retention:.4f} "
                f"grads={res.grad_evals} wall={res.wall_s:.1f}s peakMB={res.peak_mem_mb:.0f}",
                flush=True,
            )

    # aggregate
    summary = summarise(results)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, default=str))


def summarise(results: list[RunResult]):
    import collections
    by = collections.defaultdict(list)
    for r in results:
        by[r.condition].append(r)
    out = {}
    for cond, rs in by.items():
        def ms(key):
            v = [getattr(r, key) for r in rs if getattr(r, key) == getattr(r, key)]
            if not v:
                return {"mean": None, "std": None, "n": 0}
            return {
                "mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "n": len(v),
                "values": [float(x) for x in v],
            }
        out[cond] = {
            "params": rs[0].params,
            "test_acc": ms("test_acc"),
            "test_f1": ms("test_f1"),
            "test_nll": ms("test_nll"),
            "val_acc": ms("val_acc"),
            "retention": ms("retention"),
            "forgetting": ms("forgetting"),
            "grad_evals": ms("grad_evals"),
            "examples_seen": ms("examples_seen"),
            "wall_s": ms("wall_s"),
            "peak_mem_mb": ms("peak_mem_mb"),
            "generations": rs[0].generations,
        }
    return out


if __name__ == "__main__":
    main()
