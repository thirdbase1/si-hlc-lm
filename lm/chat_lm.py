#!/usr/bin/env python3
"""
chat_lm.py -- standalone inference script for a trained SI-HLC LM checkpoint.

Usage:
  python chat_lm.py --ckpt <path>                     # run the built-in demo set
  python chat_lm.py --ckpt <path> --prompt "..."      # one prompt
  python chat_lm.py --ckpt <path> --interactive       # REPL

Nothing here touches training data; this is pure generation from weights.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lm_core import (BOT_TAG, SEQ_LEN, USER_TAG, CausalLM, count_params,
                     distinct_n, generate, repetition_rate)
from tokenizers import Tokenizer

DATA_DIR = "/home/boxd/hlc/lmdata"

DEMO = [
    "Hello , how are you today ?",
    "What is your name ?",
    "Can you help me find a good restaurant ?",
    "I am feeling a little tired .",
    "What did you do last weekend ?",
    "Do you like watching movies ?",
    "Thank you , that is very helpful .",
    "Goodbye , see you later .",
]


def load(ckpt_path, data_dir=DATA_DIR):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CausalLM(**ck["config"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    tok = Tokenizer.from_file(os.path.join(data_dir, "prepared", "tokenizer.json"))
    n = count_params(model)
    assert n == 10_000_000, f"expected 10,000,000 params, got {n}"
    return model, tok, ck, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=1.15)
    args = ap.parse_args()

    model, tok, ck, n = load(args.ckpt, args.data_dir)
    device = torch.device("cpu")
    model.to(device)
    print(f"loaded {args.ckpt}")
    print(f"  condition={ck['condition']} seed={ck['seed']} params={n:,} "
          f"config={ck['config']}")

    def reply(user_text, seed=None):
        prompt = f"{USER_TAG} {user_text.strip()}\n{BOT_TAG}"
        return generate(model, tok, prompt, max_new=args.max_new,
                        temperature=args.temperature, top_k=args.top_k,
                        top_p=args.top_p,
                        repetition_penalty=args.rep_penalty, stop="\n",
                        seed=seed, device=device)

    if args.prompt:
        out = reply(args.prompt, seed=7)
        print(f"\nUser: {args.prompt}")
        print(f"Bot:  {out.strip()}")
        return

    if args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.\n")
        hist = []
        while True:
            try:
                u = input("User: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if u.lower() in ("quit", "exit", "q"):
                break
            if not u:
                continue
            out = reply(u, seed=None)
            print(f"Bot:  {out.strip()}\n")
            hist.append((u, out.strip()))
        print(f"-- {len(hist)} turns --")
        return

    # default: demo battery
    print("\n=== demo generation ===")
    for i, p in enumerate(DEMO):
        out = reply(p, seed=100 + i)
        print(f"\n[{i+1}] User: {p}")
        print(f"    Bot:  {out.strip()}")
        print(f"    (distinct-1={distinct_n(out,1):.2f} "
              f"rep3={repetition_rate(out,3):.2f})")


if __name__ == "__main__":
    main()
