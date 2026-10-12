#!/usr/bin/env python3
"""
evaluate_lm.py -- independent evaluation of a trained LM checkpoint.

Loads a checkpoint, recomputes held-out metrics from scratch, runs generation
on fixed prompts (including deliberately hard ones), and scores:
  - held-out next-token loss / perplexity
  - response coherence (Distinct-1/2, repetition rate)
  - basic question answering (token-F1 against gold answers)
  - short multi-turn conversation continuity
  - memorisation probe (exact training-dialogue recall)

The test set is used ONLY here, never for training or policy selection.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lm_core import (BOT_TAG, SEQ_LEN, USER_TAG, CausalLM, count_params,
                     distinct_n, eval_lm, format_dialogue, generate,
                     load_dailydialog, normalise, repetition_rate, token_f1)
from tokenizers import Tokenizer


def load_model_and_tok(ckpt_path, data_dir):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = CausalLM(**cfg)
    model.load_state_dict(ck["model_state"])
    model.eval()
    n = count_params(model)
    prep = os.path.join(data_dir, "prepared")
    tok = Tokenizer.from_file(os.path.join(prep, "tokenizer.json"))
    return model, tok, ck, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="/home/boxd/hlc/lmdata")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-qa", type=int, default=60)
    ap.add_argument("--n-memorise", type=int, default=30)
    args = ap.parse_args()

    model, tok, ck, nparams = load_model_and_tok(args.ckpt, args.data_dir)
    device = torch.device("cpu")
    model.to(device)
    print(f"checkpoint: {args.ckpt}")
    print(f"condition={ck['condition']} seed={ck['seed']} params={nparams:,}")
    assert nparams == 10_000_000, nparams

    prep = os.path.join(args.data_dir, "prepared")
    blocks = {}
    for s, key in [("train", "train"), ("validation", "val"), ("test", "test")]:
        blocks[key] = torch.from_numpy(
            np.load(os.path.join(prep, f"{s}_blocks.npy")))

    out = {
        "checkpoint": os.path.abspath(args.ckpt),
        "condition": ck["condition"],
        "seed": ck["seed"],
        "params": nparams,
        "config": ck["config"],
    }

    # ---- 1. held-out loss / perplexity --------------------------------
    for split in ["train", "val", "test"]:
        m = eval_lm(model, blocks[split], batch_size=8, device=device)
        out[f"{split}_loss"] = m["loss"]
        out[f"{split}_ppl"] = m["ppl"]
        print(f"{split:5s} loss={m['loss']:.4f} ppl={m['ppl']:.2f} "
              f"({m['tokens']:,} tokens)")

    # ---- 2. generation quality on fixed prompts ------------------------
    prompts = [
        "User: Hello , how are you today ?\nBot:",
        "User: What is your name ?\nBot:",
        "User: Can you help me with something ?\nBot:",
        "User: I am feeling a bit sad today .\nBot:",
        "User: What do you like to do for fun ?\nBot:",
        "User: Thank you so much for your help .\nBot:",
    ]
    gens = []
    for i, p in enumerate(prompts):
        txt = generate(model, tok, p, max_new=48, temperature=0.8, top_k=40,
                       top_p=0.9, repetition_penalty=1.15, stop="\n",
                       seed=100 + i, device=device)
        gens.append({"prompt": p, "response": txt})
        print(f"\n[gen {i}] {p.split('User:')[1].strip()}")
        print(f"   -> {txt.strip()[:200]}")

    d1 = np.mean([distinct_n(g["response"], 1) for g in gens])
    d2 = np.mean([distinct_n(g["response"], 2) for g in gens])
    rep = np.mean([repetition_rate(g["response"], 3) for g in gens])
    out["generation"] = {
        "samples": gens, "distinct_1": float(d1),
        "distinct_2": float(d2), "repetition_rate_3gram": float(rep),
    }
    print(f"\ndistinct-1={d1:.3f} distinct-2={d2:.3f} rep3={rep:.3f}")

    # ---- 3. basic question answering -----------------------------------
    # Take test dialogues whose first turn is a question (act label 2 in
    # DailyDialog); the gold answer is the next turn. Score token-F1.
    raw = load_dailydialog(args.data_dir)
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(args.data_dir, "test.parquet")).to_pydict()
    qa = []
    for di, (dlg, act) in enumerate(zip(t["dialog"], t["act"])):
        for i in range(len(dlg) - 1):
            if act[i] == 2 and i % 2 == 0:      # question in a User turn
                qa.append((dlg[i], dlg[i + 1]))
                break
    qa = qa[:args.n_qa]
    f1s, accs = [], []
    qa_detail = []
    for i, (q, gold) in enumerate(qa):
        prompt = f"{USER_TAG} {q.strip()}\n{BOT_TAG}"
        pred = generate(model, tok, prompt, max_new=32, temperature=0.7,
                        top_k=40, top_p=0.9, repetition_penalty=1.15, stop="\n",
                        seed=500 + i, device=device)
        f1 = token_f1(normalise(pred), normalise(gold))
        f1s.append(f1)
        accs.append(1.0 if normalise(pred).strip() == normalise(gold).strip() else 0.0)
        if i < 8:
            qa_detail.append({"question": q, "gold": gold,
                              "prediction": pred, "token_f1": f1})
            print(f"\n[qa {i}] Q: {q.strip()[:90]}")
            print(f"      gold: {gold.strip()[:90]}")
            print(f"      pred: {pred.strip()[:90]}  f1={f1:.3f}")
    out["qa"] = {
        "n": len(qa),
        "token_f1_mean": float(np.mean(f1s)) if f1s else 0.0,
        "exact_match": float(np.mean(accs)) if accs else 0.0,
        "examples": qa_detail,
    }
    print(f"\nQA token-F1={out['qa']['token_f1_mean']:.4f} "
          f"EM={out['qa']['exact_match']:.4f} (n={len(qa)})")

    # ---- 4. multi-turn continuity --------------------------------------
    # Seed with a real test dialogue's opening, then generate 3 turns.
    convos = []
    for di in range(6):
        dlg = t["dialog"][di]
        if len(dlg) < 3:
            continue
        transcript = f"{USER_TAG} {' '.join(dlg[0].split())}\n{BOT_TAG}"
        turns = []
        for k in range(3):
            txt = generate(model, tok, transcript, max_new=40, temperature=0.8,
                           top_k=40, top_p=0.9, repetition_penalty=1.15,
                           stop="\n", seed=900 + di * 10 + k, device=device)
            turns.append(txt.strip())
            transcript += f" {txt.strip()}\n{USER_TAG}"
        convos.append({"seed_dialogue": list(dlg[:2]), "generated": turns})
        print(f"\n[convo {di}] user: {' '.join(dlg[0].split())[:70]}")
        for k, tt in enumerate(turns):
            print(f"   bot{k+1}: {tt[:70]}")
    out["multiturn"] = convos

    # ---- 5. memorisation probe -----------------------------------------
    # Give the model the first half of a TRAIN dialogue and measure whether
    # the continuation is closer to the true continuation than a test one is.
    mem = []
    tr = pq.read_table(os.path.join(args.data_dir, "train.parquet")).to_pydict()
    for i in range(args.n_memorise):
        dlg = tr["dialog"][i]
        if len(dlg) < 4:
            continue
        cut = len(dlg) // 2
        prompt = "\n".join(
            f"{USER_TAG if j % 2 == 0 else BOT_TAG} {' '.join(x.split())}"
            for j, x in enumerate(dlg[:cut]))
        prompt += f"\n{BOT_TAG}"
        pred = generate(model, tok, prompt, max_new=40, temperature=0.6,
                        top_k=40, top_p=0.9, repetition_penalty=1.0, stop="\n",
                        seed=1300 + i, device=device)
        gold = " ".join(dlg[cut].split())
        mem.append({
            "gold_continuation": gold,
            "predicted": pred.strip(),
            "token_f1": token_f1(normalise(pred), normalise(gold)),
        })
    out["memorisation"] = {
        "n": len(mem),
        "token_f1_mean": float(np.mean([m["token_f1"] for m in mem])),
        "exact_fraction": float(np.mean(
            [1.0 if m["token_f1"] > 0.9 else 0.0 for m in mem])),
        "examples": mem[:5],
    }
    print(f"\nmemorisation token-F1={out['memorisation']['token_f1_mean']:.4f} "
          f"exact={out['memorisation']['exact_fraction']:.3f} (n={len(mem)})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
