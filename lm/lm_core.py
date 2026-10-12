#!/usr/bin/env python3
"""
lm_core.py -- causal Transformer LM, exact-10M config solver, tokenizer,
DailyDialog preparation, metrics and generation for the SI-HLC language-model
experiment.

Everything here is trained FROM SCRATCH. No pretrained weights are used.
"""
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_PARAMS = 10_000_000

# dialogue formatting constants (documented in the report)
USER_TAG = "User:"
BOT_TAG = "Bot:"
DOC_SEP = "<|doc|>"
PAD = "<|pad|>"
SPECIALS = [DOC_SEP, PAD, "<|unk|>"]

# fixed architectural ratios (documented, not tuned per condition)
D_MODEL = 320
FFN_MULT = 4
N_HEADS = 5          # 320 / 5 = 64 per head
POS_MAX = 512
SEQ_LEN = 256


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Pre-norm Transformer block (GPT-2 style)."""

    def __init__(self, d: int, nh: int, ffn: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.nh = nh

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        k = k.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        v = v.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(o)
        x = x + self.mlp(self.ln2(x))
        return x


class CausalLM(nn.Module):
    def __init__(self, vocab: int, d: int = D_MODEL, n_layers: int = 6,
                 ffn: Optional[int] = None, n_heads: int = N_HEADS,
                 pos_max: int = POS_MAX, tie: bool = True):
        super().__init__()
        ffn = ffn or d * FFN_MULT
        self.cfg = dict(vocab=vocab, d=d, n_layers=n_layers, ffn=ffn,
                        n_heads=n_heads, pos_max=pos_max, tie=tie)
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(pos_max, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, ffn) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        if tie:
            self.head.weight = self.emb.weight
        self.n_heads = n_heads

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.cfg["pos_max"], f"seq len {T} > pos_max {self.cfg['pos_max']}"
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def count_params(model: nn.Module) -> int:
    # parameters() dedupes shared tensors, so tied weights are counted once
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analytic_params(vocab: int, d: int, n_layers: int, ffn: int,
                    pos_max: int, tie: bool = True) -> int:
    """Closed form so the solver does not need to build every candidate."""
    per_layer = 3 * d * d + 3 * d        # qkv
    per_layer += d * d + d               # proj
    per_layer += 2 * (2 * d)             # two LayerNorms
    per_layer += 2 * d * ffn + ffn + d   # mlp (two linears + biases)
    total = vocab * d + pos_max * d
    total += n_layers * per_layer
    total += 2 * d                       # final LayerNorm
    if not tie:
        total += d * vocab               # untied head
    return total


def solve_exact_config(vocab: int, target: int = TARGET_PARAMS,
                       d: int = D_MODEL, pos_max: int = POS_MAX,
                       ffn_mult: int = FFN_MULT, n_heads: int = N_HEADS,
                       layers_range=range(2, 13)) -> Optional[dict]:
    """Find n_layers so the model has EXACTLY `target` trainable params."""
    ffn = d * ffn_mult
    for nl in layers_range:
        if analytic_params(vocab, d, nl, ffn, pos_max, True) == target:
            return dict(vocab=vocab, d=d, n_layers=nl, ffn=ffn,
                        n_heads=n_heads, pos_max=pos_max, tie=True)
    return None


def build_exact_model(vocab: int, seed: int = 0) -> Tuple[nn.Module, dict]:
    """Build the LM, asserting the parameter count is exactly TARGET_PARAMS."""
    cfg = solve_exact_config(vocab)
    assert cfg is not None, (
        f"no layer count gives exactly {TARGET_PARAMS} params for vocab={vocab}; "
        f"retune the tokenizer vocab size")
    torch.manual_seed(seed)
    model = CausalLM(**cfg)
    n = count_params(model)
    assert n == TARGET_PARAMS, f"param count {n} != {TARGET_PARAMS}"
    # verify the analytic form agrees with the real module
    assert analytic_params(cfg["vocab"], cfg["d"], cfg["n_layers"], cfg["ffn"],
                           cfg["pos_max"], cfg["tie"]) == n
    return model, cfg


# ---------------------------------------------------------------------------
# Data: DailyDialog
# ---------------------------------------------------------------------------
def format_dialogue(turns: List[str]) -> str:
    """Render one dialogue as an alternating User/Bot transcript."""
    out = []
    for i, t in enumerate(turns):
        tag = USER_TAG if i % 2 == 0 else BOT_TAG
        t = " ".join(t.split())           # normalise whitespace
        out.append(f"{tag} {t}")
    return "\n".join(out)


def load_dailydialog(data_dir: str) -> Dict[str, List[List[str]]]:
    import pyarrow.parquet as pq
    out = {}
    for split in ["train", "validation", "test"]:
        t = pq.read_table(os.path.join(data_dir, f"{split}.parquet")).to_pydict()
        out[split] = [list(d) for d in t["dialog"]]
    return out


@dataclass
class LMData:
    """Packed fixed-length training/eval blocks plus metadata."""
    train_blocks: torch.Tensor          # [N, SEQ_LEN] int64
    val_blocks: torch.Tensor
    test_blocks: torch.Tensor
    retention_blocks: torch.Tensor      # easy BIRTH subset, held out of curriculum
    retention_ids: np.ndarray           # block ids, for reporting
    vocab_size: int
    n_train_tokens: int
    n_train_dialogues: int


def pack_blocks(token_ids: List[List[int]], seq_len: int, vocab: int,
                doc_sep_id: int, pad_id: int) -> Tuple[torch.Tensor, np.ndarray]:
    """Concatenate tokenised dialogues (separated by DOC_SEP) and chunk into
    fixed-length blocks. Every block is exactly seq_len tokens, so token
    consumption per step is deterministic and compute is exactly matchable."""
    stream: List[int] = []
    doc_bounds: List[Tuple[int, int]] = []
    for ids in token_ids:
        start = len(stream)
        stream.extend(ids)
        stream.append(doc_sep_id)
        doc_bounds.append((start, len(stream)))
    n_blocks = len(stream) // seq_len
    if n_blocks == 0:
        raise ValueError("corpus too small for one block")
    usable = n_blocks * seq_len
    arr = np.array(stream[:usable], dtype=np.int64)
    blocks = torch.from_numpy(arr.reshape(n_blocks, seq_len))
    # map each block to the dialogues it contains (for reporting only)
    return blocks, np.array(doc_bounds, dtype=np.int64)


# ---------------------------------------------------------------------------
# Loss / evaluation
# ---------------------------------------------------------------------------
def shift_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean next-token CE over all positions (packed blocks are dense)."""
    B, T, V = logits.shape
    pred = logits[:, :-1, :].reshape(-1, V)
    gold = targets[:, 1:].reshape(-1)
    return F.cross_entropy(pred, gold, reduction="mean")


@torch.no_grad()
def eval_lm(model: nn.Module, blocks: torch.Tensor, batch_size: int = 16,
            device=None) -> Dict[str, float]:
    """Mean next-token loss and perplexity over packed blocks."""
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, len(blocks), batch_size):
        xb = blocks[i:i + batch_size]
        if device is not None:
            xb = xb.to(device)
        logits = model(xb)
        B, T, V = logits.shape
        pred = logits[:, :-1, :].reshape(-1, V)
        gold = xb[:, 1:].reshape(-1)
        loss = F.cross_entropy(pred, gold, reduction="sum")
        tot_loss += loss.item()
        tot_tok += gold.numel()
    model.train()
    mean = tot_loss / max(1, tot_tok)
    return {"loss": mean, "ppl": math.exp(min(mean, 30.0)), "tokens": tot_tok}


@torch.no_grad()
def per_block_signals(model: nn.Module, blocks: torch.Tensor,
                      batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Per-block mean loss and mean predictive entropy (uncertainty).

    CHUNKED internally: a full forward over many blocks materialises
    (N, 256, V) logits, which for N=512 and V=7618 is 4 GB before any softmax
    temporary. Chunking keeps peak memory bounded.
    """
    model.eval()
    losses, ents = [], []
    chunk = max(1, min(batch_size, 16))
    for i in range(0, len(blocks), chunk):
        xb = blocks[i:i + chunk]
        logits = model(xb)
        B, T, V = logits.shape
        pred = logits[:, :-1].reshape(-1, V)
        gold = xb[:, 1:].reshape(-1)
        lv = F.cross_entropy(pred, gold, reduction="none").view(B, T - 1)
        lp = F.log_softmax(pred, dim=-1)
        p = lp.exp()
        ent = -(p * lp).sum(-1).view(B, T - 1)
        losses.append(lv.mean(1).cpu().numpy())
        ents.append(ent.mean(1).cpu().numpy())
        del logits, lv, lp, p, ent
    model.train()
    return np.concatenate(losses), np.concatenate(ents)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model: nn.Module, tok, prompt: str, max_new: int = 60,
             temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9,
             repetition_penalty: float = 1.15, stop: Optional[str] = "\n",
             seed: Optional[int] = None, device=None) -> str:
    """Greedy/top-k/top-p sampling with a repetition penalty."""
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
    else:
        g = None
    model.eval()
    ids = tok.encode(prompt).ids
    prompt_len = len(ids)
    pos_max = model.cfg["pos_max"]
    for _ in range(max_new):
        ctx = ids[-pos_max:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()
        # repetition penalty over generated tokens only. Dividing logits by a
        # penalty > 1 can make them negative, which breaks multinomial; the
        # penalty is therefore applied as a subtraction on log-probs instead.
        if repetition_penalty != 1.0 and len(ids) > prompt_len:
            for t in set(ids[prompt_len:]):
                if logits[t] > 0:
                    logits[t] /= repetition_penalty
                else:
                    logits[t] *= repetition_penalty
        if temperature <= 0:
            nxt = int(logits.argmax().item())
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.numel()))
                logits = logits.masked_fill(logits < v[-1], float("-inf"))
            if top_p < 1.0:
                s, si = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(s, dim=-1), dim=-1)
                s = s.masked_fill(cum > top_p, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(0, si, s)
            # guard: all -inf can happen if the mask removed everything
            if not torch.isfinite(logits).any():
                logits = model(x)[0, -1].float() / temperature
            probs = F.softmax(logits, dim=-1)
            probs = probs.clamp_min(0)
            probs = probs / probs.sum()
            nxt = int(torch.multinomial(probs, 1, generator=g).item())
        ids.append(nxt)
        if stop is not None:
            tail = tok.decode(ids[-3:])
            if tail.endswith(stop) and len(ids) > prompt_len + 2:
                break
    model.train()
    return tok.decode(ids[prompt_len:])


# ---------------------------------------------------------------------------
# Text-quality metrics
# ---------------------------------------------------------------------------
def distinct_n(text: str, n: int) -> float:
    toks = text.split()
    if len(toks) < n:
        return 0.0
    ng = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(ng)) / max(1, len(ng))


def repetition_rate(text: str, n: int = 3) -> float:
    """Fraction of n-grams that are duplicates of an earlier n-gram."""
    toks = text.split()
    if len(toks) < n:
        return 0.0
    ng = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    seen, dup = set(), 0
    for g in ng:
        if g in seen:
            dup += 1
        seen.add(g)
    return dup / len(ng)


def token_f1(pred: str, gold: str) -> float:
    p, g = pred.split(), gold.split()
    if not p or not g:
        return 0.0
    common = {}
    from collections import Counter
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def normalise(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())
