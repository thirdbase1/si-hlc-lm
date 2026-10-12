#!/usr/bin/env python3
"""
run_lm_experiment.py -- driver for the SI-HLC language-model experiment.

  --mode smoke    tiny budget, 1 seed, sihlc only: verify plumbing
  --mode screen   all 4 conditions, 1 seed, short budget
  --mode main     all 4 conditions, 5 matched seeds

Compute matching: `steps` optimiser steps x `batch_size * seq_len` tokens per
step, identical for every condition. Only the RENEWAL tournaments add probe
steps, and those are recorded separately.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch

from lm_core import (SEQ_LEN, format_dialogue, load_dailydialog,
                     per_block_signals)
from lm_hlc_train import Policy, LMLHCTrainer, LMRunResult

CONDITIONS = ["adamw", "hlc_fixed", "hlc_adaptive", "sihlc"]
SEEDS = [0, 1, 2, 3, 4]


def load_prepared(data_dir: str):
    prep = os.path.join(data_dir, "prepared")
    with open(os.path.join(prep, "manifest.json")) as f:
        man = json.load(f)
    blocks = {}
    for s, key in [("train", "train"), ("validation", "val"), ("test", "test")]:
        blocks[key] = torch.from_numpy(
            np.load(os.path.join(prep, f"{s}_blocks.npy")))
    return blocks, man


def make_retention_blocks(train_blocks, retention_set, device):
    return train_blocks[retention_set].to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "screen", "main"], default="smoke")
    ap.add_argument("--data-dir", default="/home/boxd/hlc/lmdata")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"lmruns/{args.mode}_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "smoke":
        steps, gens, seeds, conds = 60, 1, [0], ["sihlc", "adamw"]
    elif args.mode == "screen":
        steps, gens, seeds, conds = 300, 2, [0], CONDITIONS
    else:
        # main: 600 steps x 16 blocks x 256 tokens = 2.46M tokens per condition.
        # Measured ~6,500 tok/s on 8 vCPU => ~6 min per condition for adamw and
        # proportionally more for the curriculum conditions (~4 h for 20 runs).
        steps, gens, seeds, conds = 600, 3, SEEDS, CONDITIONS

    if args.steps is not None:
        steps = args.steps
    if args.generations is not None:
        gens = args.generations
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    if args.conditions:
        conds = args.conditions.split(",")

    blocks, man = load_prepared(args.data_dir)
    vocab = man["vocab_size"]
    device = torch.device("cpu")

    cfg = {
        "mode": args.mode,
        "timestamp_utc": ts,
        "steps": steps,
        "generations": gens,
        "batch_size": args.batch_size,
        "seq_len": SEQ_LEN,
        "tokens_per_step": args.batch_size * SEQ_LEN,
        "token_budget": steps * args.batch_size * SEQ_LEN,
        "threads": args.threads,
        "seeds": seeds,
        "conditions": conds,
        "vocab": vocab,
        "dataset": man["dataset"],
        "license": man["license"],
        "torch": torch.__version__,
        "splits": man["splits"],
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] {json.dumps(cfg, indent=2)}", flush=True)
    print(f"[data] train blocks={len(blocks['train'])} "
          f"val={len(blocks['val'])} test={len(blocks['test'])} "
          f"vocab={vocab}", flush=True)

    results = []
    for seed in seeds:
        # retention set: the 200 easiest TRAIN blocks, scored on a random
        # SAMPLE (not all 6109) by a model seeded identically to this run, so
        # the subset is identical across conditions for a given seed and never
        # touches validation or test data. Scoring all blocks with an
        # untrained model costs minutes and buys nothing at init.
        torch.manual_seed(seed)
        tmp, _ = __import__("lm_core").build_exact_model(vocab, seed=seed)
        rng = np.random.default_rng(seed + 77)
        sample = rng.choice(len(blocks["train"]), size=600, replace=False)
        d, _ = per_block_signals(tmp, blocks["train"][sample], batch_size=8)
        del tmp
        retention_idx = sample[np.argsort(d)[:200]]
        retention_blocks = blocks["train"][retention_idx]
        np.save(os.path.join(out_dir, f"retention_idx_seed{seed}.npy"),
                retention_idx)

        for cond in conds:
            tag = f"{cond}_seed{seed}"
            print(f"\n===== {tag} =====", flush=True)
            policy = Policy()
            trainer = LMLHCTrainer(
                condition=cond, seed=seed, blocks=blocks,
                retention_blocks=retention_blocks, vocab=vocab, policy=policy,
                batch_size=args.batch_size, seq_len=SEQ_LEN,
                threads=args.threads, out_dir=out_dir, max_generations=gens,
            )
            assert trainer.params == 10_000_000, trainer.params
            t0 = time.time()
            res = trainer.run(total_steps=steps)
            res.extra["driver_wall_s"] = time.time() - t0

            ckpt = {
                "condition": cond, "seed": seed,
                "model_state": trainer.model.state_dict(),
                "opt_state": trainer.opt.state_dict(),
                "params": trainer.params, "config": trainer.cfg,
                "policy": trainer.policy.to_dict(),
                "result": res.__dict__, "cfg": cfg,
            }
            torch.save(ckpt, os.path.join(out_dir, f"ckpt_{tag}.pt"))
            with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as f:
                json.dump(res.__dict__, f, indent=2, default=str)

            results.append(res)
            print(f"[{tag}] test_loss={res.test_loss:.4f} "
                  f"test_ppl={res.test_ppl:.2f} val_ppl={res.val_ppl:.2f} "
                  f"steps={res.steps} probe={res.probe_steps} "
                  f"tokens={res.tokens_seen:,} wall={res.wall_s:.0f}s "
                  f"peakMB={res.peak_mem_mb:.0f}", flush=True)

    summary = summarise(results)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, default=str))


def summarise(results):
    import collections
    by = collections.defaultdict(list)
    for r in results:
        by[r.condition].append(r)

    def ms(rs, key):
        v = [getattr(r, key) for r in rs
             if getattr(r, key) == getattr(r, key)]
        if not v:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "n": len(v), "values": [float(x) for x in v]}

    out = {}
    for cond, rs in by.items():
        out[cond] = {
            "params": rs[0].params,
            "config": rs[0].config,
            "test_loss": ms(rs, "test_loss"),
            "test_ppl": ms(rs, "test_ppl"),
            "val_loss": ms(rs, "val_loss"),
            "val_ppl": ms(rs, "val_ppl"),
            "retention_ppl": ms(rs, "retention_ppl"),
            "steps": ms(rs, "steps"),
            "probe_steps": ms(rs, "probe_steps"),
            "tokens_seen": ms(rs, "tokens_seen"),
            "wall_s": ms(rs, "wall_s"),
            "peak_mem_mb": ms(rs, "peak_mem_mb"),
            "generations": rs[0].generations,
        }
    return out


if __name__ == "__main__":
    main()
