#!/usr/bin/env python3
"""
lm_hlc_train.py -- SI-HLC applied to causal language-model training.

The six life-cycle stages are preserved from the MNIST implementation, with the
sampling primitives re-expressed for packed token blocks. The mechanisms are NOT
invented: each stage keeps its documented role and its measurable transition rule.

  BIRTH        learn from low-loss (foundational) blocks
  GROWTH       gradually widen from easy to the full distribution
  EXPLORATION  prioritise high-entropy (uncertain) blocks
  REFLECTION   revisit blocks the model repeatedly gets wrong
  CONSOLIDATION mix replay blocks with new material (anti-forgetting)
  RENEWAL      evaluate progress/retention/stagnation; (D) run the policy tournament

Conditions:
  A. adamw        - standard AdamW, uniform shuffled blocks, fixed LR
  B. hlc_fixed    - six-stage curriculum, transitions on STEP COUNT only
  C. hlc_adaptive - six stages, transitions driven by experience signals
  D. sihlc        - C + generational policy selection at RENEWAL

Compute matching: every condition performs EXACTLY `total_steps` optimiser steps
on exactly `batch_size * seq_len` tokens per step, so the token budget is
identical. Probe (tournament) steps are counted separately as probe_steps and are
NOT part of the training budget.
"""
import copy
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lm_core import (CausalLM, TARGET_PARAMS, build_exact_model, count_params,
                     eval_lm, per_block_signals)

STAGES = ["BIRTH", "GROWTH", "EXPLORATION", "REFLECTION", "CONSOLIDATION", "RENEWAL"]
ACTIVE_STAGES = STAGES[:5]


# ---------------------------------------------------------------------------
# Policy: the object generational selection evolves
# ---------------------------------------------------------------------------
@dataclass
class Policy:
    # --- transition thresholds (measurable) -------------------------------
    # Loss scale for this LM: untrained loss is ~ln(V)=8.94; a trained model
    # reaches ~4.5-5.0 on validation. Thresholds are set on that scale.
    # BIRTH -> GROWTH: mean block loss must fall below this.
    birth_to_growth_loss: float = 6.5
    # GROWTH -> EXPLORATION: mean block loss below this.
    growth_to_expl_loss: float = 5.0
    # EXPLORATION -> REFLECTION: learning progress (loss delta over a window)
    # must fall below this -- i.e. uncertain blocks stopped helping.
    expl_to_refl_progress: float = 0.01
    # REFLECTION -> CONSOLIDATION: retention on the BIRTH subset must be at
    # least this good. Measured as a LOSS (lower is better), so the transition
    # fires when retention loss has fallen to `refl_to_cons_retention` or
    # below -- i.e. the easy foundation is still learned.
    refl_to_cons_retention: float = 5.0
    # CONSOLIDATION -> RENEWAL: stagnation (relative loss change) below this.
    cons_to_renew_stag: float = 0.02
    # --- stage durations (optimiser steps) --------------------------------
    d_birth: int = 120
    d_growth: int = 180
    d_exploration: int = 200
    d_reflection: int = 180
    d_consolidation: int = 180
    d_renewal: int = 0          # evaluation only: consumes no training steps
    # --- sampling proportions ---------------------------------------------
    p_hard: float = 0.5         # GROWTH: fraction sampled difficulty-weighted
    p_uncertain: float = 0.5    # EXPLORATION: fraction entropy-weighted
    p_missed: float = 0.7       # REFLECTION: fraction miss-weighted
    p_replay: float = 0.3       # CONSOLIDATION: fraction from the replay pool
    birth_easy_frac: float = 0.25   # BIRTH: easiest fraction of blocks
    growth_easy_frac: float = 0.30  # GROWTH: easy fraction at the start
    # --- optimiser ---------------------------------------------------------
    lr: float = 3e-4
    warmup_frac: float = 0.05     # warmup as a FRACTION of the total budget,
                                  # so it scales instead of being a fixed step
                                  # count that swamps short runs
    # minimum steps a stage must run before its transition rule may fire
    min_stage_steps: int = 20

    def warmup_steps(self, total_steps: int) -> int:
        """Warmup length derived from the run's own step budget."""
        return max(1, int(round(total_steps * self.warmup_frac)))

    def to_dict(self):
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d):
        p = Policy()
        for k, v in d.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p

    def clone(self):
        return Policy.from_dict(self.to_dict())


def mutate(policy: Policy, rng: np.random.Generator) -> Policy:
    """Perturb thresholds, durations, proportions and LR (clamped)."""
    c = policy.clone()

    def jit(v, lo, hi, frac=0.2):
        span = (hi - lo) * frac
        return float(np.clip(v + rng.uniform(-span, span), lo, hi))

    c.birth_to_growth_loss = jit(c.birth_to_growth_loss, 3.0, 8.5)
    c.growth_to_expl_loss = jit(c.growth_to_expl_loss, 2.5, 7.0)
    c.expl_to_refl_progress = jit(c.expl_to_refl_progress, 1e-4, 5e-2)
    c.refl_to_cons_retention = jit(c.refl_to_cons_retention, 3.5, 6.5)
    c.cons_to_renew_stag = jit(c.cons_to_renew_stag, 0.002, 0.10)
    for name in ["d_birth", "d_growth", "d_exploration",
                 "d_reflection", "d_consolidation"]:
        v = getattr(c, name)
        setattr(c, name, int(np.clip(round(v * rng.uniform(0.6, 1.5)), 30, 600)))
    c.p_hard = jit(c.p_hard, 0.20, 0.90)
    c.p_uncertain = jit(c.p_uncertain, 0.20, 0.90)
    c.p_missed = jit(c.p_missed, 0.30, 0.95)
    c.p_replay = jit(c.p_replay, 0.05, 0.80)
    c.birth_easy_frac = jit(c.birth_easy_frac, 0.05, 0.50)
    c.growth_easy_frac = jit(c.growth_easy_frac, 0.10, 0.60)
    c.lr = float(np.clip(c.lr * rng.uniform(0.5, 2.0), 5e-5, 2e-3))
    c.warmup_frac = float(np.clip(c.warmup_frac * rng.uniform(0.5, 2.0), 0.01, 0.20))
    return c


# ---------------------------------------------------------------------------
# Experience record
# ---------------------------------------------------------------------------
@dataclass
class Experience:
    """Rolling record of learning signals driving stage decisions."""
    loss_window: List[float] = field(default_factory=list)
    val_history: List[Dict[str, float]] = field(default_factory=list)
    miss_counts: Optional[np.ndarray] = None      # EMA of per-block error
    difficulty: Optional[np.ndarray] = None       # per-block mean CE
    uncertainty: Optional[np.ndarray] = None      # per-block mean entropy
    seen: Optional[np.ndarray] = None
    replay_pool: List[int] = field(default_factory=list)
    retention_score: float = float("nan")
    steps: int = 0
    tokens: int = 0
    probe_steps: int = 0

    def recent_loss(self, k: int = 30) -> float:
        if not self.loss_window:
            return float("nan")
        return float(np.mean(self.loss_window[-k:]))

    def learning_progress(self, k: int = 30) -> float:
        """Positive when the recent window is below the earlier one."""
        if len(self.loss_window) < 2 * k:
            return 0.0
        early = float(np.mean(self.loss_window[-2 * k:-k]))
        late = float(np.mean(self.loss_window[-k:]))
        return early - late

    def stagnation(self, k: int = 50) -> float:
        if len(self.loss_window) < 2 * k:
            return 1.0
        a = float(np.mean(self.loss_window[-2 * k:-k]))
        b = float(np.mean(self.loss_window[-k:]))
        return abs(a - b) / (abs(a) + 1e-8)

    def record_step(self, loss: float, bs: int, seq_len: int):
        self.loss_window.append(float(loss))
        self.steps += 1
        self.tokens += bs * seq_len
        if len(self.loss_window) > 3000:
            self.loss_window = self.loss_window[-3000:]

    def unrecord_step(self, n: int, bs: int, seq_len: int):
        """Roll back bookkeeping for steps that were exploratory probes.

        The lineage's step and token counts must reflect ONLY the steps that
        actually advanced the selected branch, otherwise the generational
        condition looks like it consumed a bigger compute budget than the
        baselines. Probe cost is reported separately (probe_steps /
        probe_tokens) so the comparison stays honest.
        """
        self.steps -= n
        self.tokens -= n * bs * seq_len


# ---------------------------------------------------------------------------
# Sampler over packed blocks
# ---------------------------------------------------------------------------
class BlockSampler:
    def __init__(self, n: int, seed: int):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(n)
        self.ptr = 0

    def next_uniform(self, k: int) -> np.ndarray:
        """Uniform shuffled minibatches, reshuffling when exhausted."""
        out = []
        while len(out) < k:
            if self.ptr >= self.n:
                self.order = self.rng.permutation(self.n)
                self.ptr = 0
            take = min(k - len(out), self.n - self.ptr)
            out.extend(self.order[self.ptr:self.ptr + take].tolist())
            self.ptr += take
        return np.asarray(out)

    def next_weighted(self, k: int, weights: np.ndarray) -> np.ndarray:
        """Sample k blocks with probability proportional to `weights`.

        Uses inverse-CDF sampling with a precomputed normalisation instead of
        np.random.choice(p=...), which rebuilds a cumulative distribution over
        the whole array on every call and is orders of magnitude slower.
        """
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 1e-8, None)
        cum = np.cumsum(w / w.sum())
        r = self.rng.random(k)
        return np.searchsorted(cum, r, side="right").clip(0, self.n - 1)

    def next_from_pool(self, k: int, pool: List[int]) -> np.ndarray:
        if not pool:
            return self.next_uniform(k)
        # uniform draw from the pool; the pool is already a large random sample
        # of seen blocks, so direct indexing is equivalent to choice() but
        # avoids building a distribution each call
        m = len(pool)
        pos = self.rng.integers(0, m, size=k)
        return np.asarray([pool[i] for i in pos], dtype=np.int64)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class LMRunResult:
    condition: str
    seed: int
    params: int
    config: Dict
    test_loss: float = 0.0
    test_ppl: float = 0.0
    val_loss: float = 0.0
    val_ppl: float = 0.0
    train_loss_final: float = 0.0
    retention_ppl: float = float("nan")
    retention_loss: float = float("nan")
    steps: int = 0
    probe_steps: int = 0
    probe_tokens: int = 0
    tokens_seen: int = 0
    wall_s: float = 0.0
    peak_mem_mb: float = 0.0
    stage_log: List[Dict] = field(default_factory=list)
    generations: List[Dict] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class LMLHCTrainer:
    def __init__(self, condition: str, seed: int, blocks: Dict[str, torch.Tensor],
                 retention_blocks: Optional[torch.Tensor], vocab: int,
                 policy: Policy, batch_size: int = 16, seq_len: int = 256,
                 threads: int = 8, out_dir: str = ".", max_generations: int = 3,
                 tokenizer=None):
        assert condition in ("adamw", "hlc_fixed", "hlc_adaptive", "sihlc")
        self.condition = condition
        self.seed = seed
        self.policy = policy
        self.bs = batch_size
        self.seq_len = seq_len
        self.out_dir = out_dir
        self.threads = threads
        self.max_generations = max_generations
        self.tok = tokenizer
        torch.set_num_threads(threads)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cpu")
        self.blocks = blocks
        self.retention_blocks = retention_blocks

        self.model, self.cfg = build_exact_model(vocab, seed=seed)
        self.params = count_params(self.model)
        assert self.params == TARGET_PARAMS, f"{self.params} != {TARGET_PARAMS}"
        self.model.to(self.device)

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=policy.lr, weight_decay=0.01,
            betas=(0.9, 0.95))

        self.sampler = BlockSampler(len(blocks["train"]), seed + 1)
        n_tr = len(blocks["train"])
        self.exp = Experience(
            miss_counts=np.zeros(n_tr, dtype=np.float32),
            difficulty=np.zeros(n_tr, dtype=np.float32),
            uncertainty=np.zeros(n_tr, dtype=np.float32),
            seen=np.zeros(n_tr, dtype=bool),
        )
        self.easy_pool: List[int] = []
        self.stage_log: List[Dict] = []
        self.generations: List[Dict] = []
        self.t0 = None
        self.peak_mem = 0.0
        self._lr_sched = None

    # -- helpers ------------------------------------------------------------
    def _mem(self):
        """Track peak RSS. Reads VmHWM from /proc (the kernel high-water
        mark). Sampled every 25 calls -- opening /proc each step is itself
        a measurable cost."""
        self._mem_calls = getattr(self, "_mem_calls", 0) + 1
        if self._mem_calls % 25:
            return
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        kb = float(line.split()[1])
                        self.peak_mem = max(self.peak_mem, kb / 1024.0)
                        return
        except Exception:
            pass

    def _lr_at(self, step: int, total: int) -> float:
        """Linear warmup then cosine decay -- identical for every condition."""
        p = self.policy
        warm = p.warmup_steps(total)
        if step < warm:
            return p.lr * (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        t = min(max(t, 0.0), 1.0)
        return p.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    def _set_lr(self, lr: float):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def _step(self, idx: np.ndarray, total: int) -> float:
        xb = self.blocks["train"][idx].to(self.device)
        self._set_lr(self._lr_at(self.exp.steps, total))
        self.opt.zero_grad(set_to_none=True)
        logits = self.model(xb)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               xb[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return float(loss.item())

    def _refresh_signals(self, idx: np.ndarray):
        """Update difficulty / uncertainty / miss counts for these blocks.

        The forward pass is CHUNKED: a full-batch forward on 512 blocks
        materialises (512, 256, 7618) logits = 4 GB, and the softmax + entropy
        temporaries on top of that peak near 20 GB RSS. Chunking bounds peak
        memory to a few hundred MB regardless of how many blocks are scored.
        """
        xb_all = self.blocks["train"][idx]
        n = len(idx)
        chunk = 16                      # blocks per forward pass
        losses = np.empty(n, dtype=np.float32)
        ents = np.empty(n, dtype=np.float32)
        accs = np.empty(n, dtype=np.float32)
        for i in range(0, n, chunk):
            xb = xb_all[i:i + chunk].to(self.device)
            with torch.no_grad():
                logits = self.model(xb)
                B, T, V = logits.shape
                pred = logits[:, :-1].reshape(-1, V)
                gold = xb[:, 1:].reshape(-1)
                lv = F.cross_entropy(pred, gold, reduction="none").view(B, T - 1)
                lp = F.log_softmax(pred, dim=-1)
                p = lp.exp()
                # entropy = -sum p log p, computed from log-probs to avoid a
                # second full-size temporary
                ent = -(p * lp).sum(-1).view(B, T - 1)
                correct = (pred.argmax(-1) == gold).float().view(B, T - 1).mean(1)
            m = len(xb)
            losses[i:i + m] = lv.mean(1).cpu().numpy()
            ents[i:i + m] = ent.mean(1).cpu().numpy()
            accs[i:i + m] = correct.cpu().numpy()
            del logits, lv, lp, p, ent

        self.exp.difficulty[idx] = losses
        self.exp.uncertainty[idx] = ents
        wrong = 1.0 - accs
        self.exp.miss_counts[idx] = (self.exp.miss_counts[idx] * 0.9 + wrong * 0.1)
        self.exp.seen[idx] = True
        self.exp.replay_pool.extend(int(i) for i in idx)
        if len(self.exp.replay_pool) > 20000:
            self.exp.replay_pool = self.exp.replay_pool[-20000:]

    def _retention(self) -> float:
        """Loss on the BIRTH (easy) subset -- lower is better.

        Uses at most 64 blocks: the signal is a relative forgetting measure,
        and 64 blocks x 256 tokens = 16k tokens is plenty of resolution while
        costing a third of the full 200-block pass.
        """
        if self.retention_blocks is None:
            return float("nan")
        n = min(64, len(self.retention_blocks))
        m = eval_lm(self.model, self.retention_blocks[:n], batch_size=8,
                    device=self.device)
        return m["loss"]

    def _log_stage(self, stage: str, note: str = ""):
        self.stage_log.append({
            "stage": stage,
            "step": self.exp.steps,
            "tokens": self.exp.tokens,
            "train_loss": self.exp.recent_loss(),
            "retention_loss": self._retention(),
            "note": note,
        })
    # -- the six stages ----------------------------------------------------
    def stage_birth(self, budget: int, hold: bool = False, total: int = 0) -> str:
        """Learn foundational patterns from the easiest blocks."""
        for s in range(budget):
            if s == 0 and not self.easy_pool:
                if np.all(self.exp.difficulty == 0):
                    # Score a random sample of blocks with the current model and
                    # use only that sample to define "easy". Blocks that were
                    # NOT scored are excluded from the easy pool rather than
                    # treated as difficulty 0 (which would make every unscored
                    # block look trivially easy).
                    sample = self.sampler.next_uniform(
                        min(512, len(self.exp.difficulty)))
                    self._refresh_signals(sample)
                    d = self.exp.difficulty[sample]
                    thr = np.quantile(d, max(self.policy.birth_easy_frac, 1e-6))
                    self.easy_pool = sample[d <= thr].tolist()
                    self.easy_sample = sample          # remember what was scored
                else:
                    thr = np.quantile(
                        self.exp.difficulty[self.exp.difficulty > 0],
                        max(self.policy.birth_easy_frac, 1e-6))
                    self.easy_pool = np.where(
                        (self.exp.difficulty > 0)
                        & (self.exp.difficulty <= thr))[0].tolist()
            if self.easy_pool:
                idx = self.sampler.next_from_pool(self.bs, self.easy_pool)
            else:
                idx = self.sampler.next_uniform(self.bs)
            l = self._step(idx, total)
            self.exp.record_step(l, self.bs, self.seq_len)
            if s % 5 == 0:
                self._refresh_signals(idx)
            self._mem()
            if (not hold and s >= self.policy.min_stage_steps
                    and self.exp.recent_loss(30) <= self.policy.birth_to_growth_loss):
                return "GROWTH"
        return "GROWTH"

    def stage_growth(self, budget: int, hold: bool = False, total: int = 0) -> str:
        """Gradually widen from the easy subset toward the full distribution."""
        for s in range(budget):
            frac = self.policy.growth_easy_frac + (1.0 - self.policy.growth_easy_frac) \
                * min(1.0, s / max(1, budget))
            n_easy = int(self.bs * frac)
            n_rest = self.bs - n_easy
            idx = np.concatenate([
                self.sampler.next_from_pool(n_easy, self.easy_pool) if n_easy else
                np.empty(0, dtype=np.int64),
                self.sampler.next_weighted(n_rest, self.exp.difficulty + 1e-6)
                if n_rest else np.empty(0, dtype=np.int64),
            ]).astype(np.int64)
            if len(idx) < self.bs:
                idx = np.concatenate(
                    [idx, self.sampler.next_uniform(self.bs - len(idx))])
            l = self._step(idx, total)
            self.exp.record_step(l, self.bs, self.seq_len)
            if s % 5 == 0:
                self._refresh_signals(idx)
            self._mem()
            if (not hold and s >= self.policy.min_stage_steps
                    and self.exp.recent_loss(30) <= self.policy.growth_to_expl_loss):
                return "EXPLORATION"
        return "EXPLORATION"

    def stage_exploration(self, budget: int, hold: bool = False,
                          total: int = 0) -> str:
        """Prioritise uncertain (high-entropy) blocks."""
        for s in range(budget):
            n_unc = int(self.bs * self.policy.p_uncertain)
            n_rand = self.bs - n_unc
            idx = np.concatenate([
                self.sampler.next_weighted(n_unc, self.exp.uncertainty + 1e-6),
                self.sampler.next_uniform(n_rand),
            ]).astype(np.int64)
            l = self._step(idx, total)
            self.exp.record_step(l, self.bs, self.seq_len)
            if s % 5 == 0:
                self._refresh_signals(idx)
            self._mem()
            if (not hold and s >= self.policy.min_stage_steps
                    and self.exp.learning_progress(30)
                    < self.policy.expl_to_refl_progress):
                return "REFLECTION"
        return "REFLECTION"

    def stage_reflection(self, budget: int, hold: bool = False,
                         total: int = 0) -> str:
        """Revisit blocks the model repeatedly gets wrong."""
        for s in range(budget):
            n_miss = int(self.bs * self.policy.p_missed)
            n_rand = self.bs - n_miss
            idx = np.concatenate([
                self.sampler.next_weighted(n_miss, self.exp.miss_counts + 1e-6),
                self.sampler.next_uniform(n_rand),
            ]).astype(np.int64)
            l = self._step(idx, total)
            self.exp.record_step(l, self.bs, self.seq_len)
            if s % 5 == 0:
                self._refresh_signals(idx)
            self._mem()
            # retention is a full forward pass over the BIRTH subset, so it is
            # only evaluated periodically -- never every step
            if (not hold and s >= self.policy.min_stage_steps and s % 10 == 0
                    and self._retention() <= self.policy.refl_to_cons_retention):
                return "CONSOLIDATION"
        return "CONSOLIDATION"

    def stage_consolidation(self, budget: int, hold: bool = False,
                            total: int = 0) -> str:
        """Mix replay blocks with new material to reduce forgetting."""
        for s in range(budget):
            n_old = int(self.bs * self.policy.p_replay)
            n_new = self.bs - n_old
            idx = np.concatenate([
                self.sampler.next_from_pool(n_old, self.exp.replay_pool),
                self.sampler.next_uniform(n_new),
            ]).astype(np.int64)
            l = self._step(idx, total)
            self.exp.record_step(l, self.bs, self.seq_len)
            if s % 5 == 0:
                self._refresh_signals(idx)
            self._mem()
            if (not hold and s >= self.policy.min_stage_steps
                    and self.exp.stagnation(50) < self.policy.cons_to_renew_stag):
                return "RENEWAL"
        return "RENEWAL"

    def stage_renewal(self, generation: int, total: int) -> Optional[Policy]:
        """Evaluate progress/retention/stagnation; run the policy tournament."""
        ret = self._retention()
        self._log_stage("RENEWAL",
                        f"gen={generation} stagnation={self.exp.stagnation(50):.5f} "
                        f"retention_loss={ret:.4f}")
        if self.condition != "sihlc":
            return None

        # ---- generational selection from the SAME checkpoint ------------
        model_state = copy.deepcopy(self.model.state_dict())
        opt_state = copy.deepcopy(self.opt.state_dict())
        incumbent = self.policy.clone()
        base_val = eval_lm(self.model, self.blocks["val"], batch_size=8,
                           device=self.device)

        rng = np.random.default_rng(self.seed * 1000 + generation)
        candidates = [mutate(incumbent, rng) for _ in range(3)]
        probe_budget = max(10, min(30, self.policy.d_consolidation // 6))

        results = []
        for ci, cand in enumerate(candidates):
            self.model.load_state_dict(model_state)
            self.opt.load_state_dict(opt_state)
            self._set_lr(cand.lr)
            before = self.exp.steps
            for s in range(probe_budget):
                n_old = int(self.bs * cand.p_replay)
                idx = np.concatenate([
                    self.sampler.next_from_pool(n_old, self.exp.replay_pool),
                    self.sampler.next_uniform(self.bs - n_old),
                ]).astype(np.int64)
                l = self._step(idx, total)
                self.exp.record_step(l, self.bs, self.seq_len)
            cand_val = eval_lm(self.model, self.blocks["val"], batch_size=8,
                               device=self.device)
            used = self.exp.steps - before
            results.append({
                "candidate": ci,
                "policy": cand.to_dict(),
                "val_loss": cand_val["loss"],
                "val_ppl": cand_val["ppl"],
                "probe_steps": int(used),
            })
            # probe steps are exploratory and tracked separately; the lineage's
            # step and token counts are rolled back so the generational
            # condition is compute-matched to the baselines
            self.exp.probe_steps += used
            self.exp.unrecord_step(used, self.bs, self.seq_len)
            # restore the loss window (the lineage's experience is unchanged)
            self.exp.loss_window = self.exp.loss_window[:len(self.exp.loss_window) - used]
            self.exp.steps = before

        best = min(results, key=lambda r: r["val_loss"])
        accepted = best["val_loss"] < base_val["loss"]

        self.generations.append({
            "generation": generation,
            "step": before,
            "incumbent_val_loss": base_val["loss"],
            "incumbent_val_ppl": base_val["ppl"],
            "candidates": results,
            "accepted": accepted,
            "selected": "candidate" if accepted else "incumbent",
            "selected_val_loss": best["val_loss"] if accepted else base_val["loss"],
        })

        # the lineage checkpoint is always restored; a rejected candidate
        # never advances it
        self.model.load_state_dict(model_state)
        self.opt.load_state_dict(opt_state)
        if accepted:
            return Policy.from_dict(best["policy"])
        return incumbent.clone()

    # -- main loops --------------------------------------------------------
    def run_adamw(self, total_steps: int):
        """Condition A: plain AdamW with warmup+cosine, uniform shuffled blocks."""
        self._log_stage("ADAMW", "baseline: uniform shuffled blocks, fixed schedule")
        for s in range(total_steps):
            idx = self.sampler.next_uniform(self.bs)
            l = self._step(idx, total_steps)
            self.exp.record_step(l, self.bs, self.seq_len)
            if s % 20 == 0:
                self._refresh_signals(idx)
            self._mem()

    def run_fixed(self, total_steps: int):
        """Condition B: transitions purely on step count."""
        raw = [self.policy.d_birth, self.policy.d_growth,
               self.policy.d_exploration, self.policy.d_reflection,
               self.policy.d_consolidation]
        tot = sum(raw)
        plan = [max(1, round(b * total_steps / tot)) for b in raw]
        plan[0] += total_steps - sum(plan)
        fns = {
            "BIRTH": self.stage_birth, "GROWTH": self.stage_growth,
            "EXPLORATION": self.stage_exploration, "REFLECTION": self.stage_reflection,
            "CONSOLIDATION": self.stage_consolidation,
        }
        for name, budget in zip(ACTIVE_STAGES, plan):
            self._log_stage(name, f"fixed schedule budget={budget}")
            fns[name](budget, hold=True, total=total_steps)

    def run_adaptive(self, total_steps: int):
        """Conditions C and D: signal-driven transitions, surplus carried forward."""
        n_gens = 1 if self.condition == "hlc_adaptive" else self.max_generations
        per_gen = max(1, total_steps // n_gens)
        stage_fns = [("BIRTH", self.stage_birth), ("GROWTH", self.stage_growth),
                     ("EXPLORATION", self.stage_exploration),
                     ("REFLECTION", self.stage_reflection),
                     ("CONSOLIDATION", self.stage_consolidation)]
        for gen in range(n_gens):
            remaining = total_steps - self.exp.steps
            if remaining <= 0:
                break
            gen_budget = min(per_gen, remaining)
            gen_start = self.exp.steps
            weights = [self.policy.d_birth, self.policy.d_growth,
                       self.policy.d_exploration, self.policy.d_reflection,
                       self.policy.d_consolidation]
            tot = sum(weights)
            planned = [max(1, round(w * gen_budget / tot)) for w in weights]
            planned[0] += gen_budget - sum(planned)

            carried = 0
            for (name, fn), want0 in zip(stage_fns, planned):
                want = want0 + carried
                before = self.exp.steps
                self._log_stage(name, f"gen={gen} want={want}")
                fn(want, total=total_steps)
                used = self.exp.steps - before
                carried = max(0, want - used)

            guard = 0
            while carried > 0 and guard < 10:
                before = self.exp.steps
                self.stage_consolidation(carried, total=total_steps)
                used = self.exp.steps - before
                if used <= 0:
                    break
                carried = max(0, carried - used)
                guard += 1

            short = gen_budget - (self.exp.steps - gen_start)
            if short > 0:
                self.stage_consolidation(short, hold=True, total=total_steps)

            new_policy = self.stage_renewal(gen, total_steps)
            if new_policy is not None:
                self.policy = new_policy
            if self.condition == "hlc_adaptive":
                break

    def run(self, total_steps: int) -> LMRunResult:
        self.t0 = time.time()
        if self.condition == "adamw":
            self.run_adamw(total_steps)
        elif self.condition == "hlc_fixed":
            self.run_fixed(total_steps)
        else:
            self.run_adaptive(total_steps)
        wall = time.time() - self.t0
        self._mem()

        val = eval_lm(self.model, self.blocks["val"], batch_size=8,
                      device=self.device)
        test = eval_lm(self.model, self.blocks["test"], batch_size=8,
                       device=self.device)
        ret = self._retention()

        return LMRunResult(
            condition=self.condition, seed=self.seed, params=self.params,
            config=self.cfg,
            test_loss=test["loss"], test_ppl=test["ppl"],
            val_loss=val["loss"], val_ppl=val["ppl"],
            train_loss_final=self.exp.recent_loss(50),
            retention_ppl=math.exp(min(ret, 30.0)) if ret == ret else float("nan"),
            retention_loss=ret,
            steps=self.exp.steps, probe_steps=self.exp.probe_steps,
            probe_tokens=self.exp.probe_steps * self.bs * self.seq_len,
            tokens_seen=self.exp.tokens, wall_s=wall, peak_mem_mb=self.peak_mem,
            stage_log=self.stage_log, generations=self.generations,
        )
