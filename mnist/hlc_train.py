#!/usr/bin/env python3
"""
SI-HLC trainer: six-stage life-cycle curriculum + generational policy selection.

Conditions:
  A. adamw        - standard AdamW, shuffled uniform sampling, fixed LR
  B. hlc_fixed    - six-stage curriculum, transitions on STEP COUNT only
  C. hlc_adaptive - six stages, transitions driven by experience signals
  D. sihlc        - C + generational policy evolution at RENEWAL
"""
import copy
import json
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

from hlc_core import (
    Experience,
    STAGES,
    TARGET_PARAMS,
    build_model,
    count_params,
    evaluate,
    init_weights,
    predict_with_uncertainty,
    score_difficulty,
    split_train_val,
    to_tensor,
)


# ----------------------------------------------------------------------------
# Policy: the tunable object that generational selection evolves
# ----------------------------------------------------------------------------

@dataclass
class Policy:
    # --- transition thresholds -------------------------------------------
    # BIRTH -> GROWTH: only once the easy-example error is low enough that the
    # foundation is genuinely learned (untrained err is ~0.9, so 0.35 is a real
    # gate rather than something satisfied at initialisation).
    birth_to_growth_acc: float = 0.65
    # GROWTH -> EXPLORATION: error on the harder mix must drop below this.
    growth_to_expl_err: float = 0.12
    # EXPLORATION -> REFLECTION: learning progress (loss delta over a window)
    # must fall below this, i.e. the model has stopped improving on uncertain
    # examples. Requires a full window, so it cannot fire at step 1.
    expl_to_refl_progress: float = 5e-4
    # REFLECTION -> CONSOLIDATION: retention on the easy foundation must be at
    # least this before we stop drilling the missed examples.
    refl_to_cons_retention: float = 0.90
    # CONSOLIDATION -> RENEWAL: stagnation (relative loss change) below this.
    cons_to_renew_stag: float = 0.01
    # --- stage durations (gradient steps) --------------------------------
    d_birth: int = 150
    d_growth: int = 250
    d_exploration: int = 300
    d_reflection: int = 250
    d_consolidation: int = 250
    d_renewal: int = 100
    # --- sampling proportions --------------------------------------------
    p_replay: float = 0.30
    p_uncertain: float = 0.50
    p_hard: float = 0.50
    p_missed: float = 0.70
    growth_easy_frac: float = 0.30
    # --- optimiser --------------------------------------------------------
    lr: float = 1e-3
    # minimum steps a stage must run before its transition rule may fire
    min_stage_steps: int = 25

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
    c = policy.clone()

    def jit(v, lo, hi, frac=0.2):
        span = (hi - lo) * frac
        return float(np.clip(v + rng.uniform(-span, span), lo, hi))

    c.birth_to_growth_acc = jit(c.birth_to_growth_acc, 0.80, 0.99)
    c.growth_to_expl_err = jit(c.growth_to_expl_err, 0.02, 0.30)
    c.expl_to_refl_progress = jit(c.expl_to_refl_progress, 1e-5, 1e-2)
    c.refl_to_cons_retention = jit(c.refl_to_cons_retention, 0.85, 0.995)
    c.cons_to_renew_stag = jit(c.cons_to_renew_stag, 0.002, 0.10)
    for name in ["d_birth", "d_growth", "d_exploration",
                 "d_reflection", "d_consolidation", "d_renewal"]:
        v = getattr(c, name)
        setattr(c, name, int(np.clip(round(v * rng.uniform(0.6, 1.5)), 40, 600)))
    c.p_replay = jit(c.p_replay, 0.05, 0.80)
    c.p_uncertain = jit(c.p_uncertain, 0.20, 0.90)
    c.p_hard = jit(c.p_hard, 0.20, 0.90)
    c.p_missed = jit(c.p_missed, 0.30, 0.95)
    c.growth_easy_frac = jit(c.growth_easy_frac, 0.10, 0.60)
    c.lr = float(np.clip(c.lr * rng.uniform(0.5, 2.0), 1e-4, 5e-3))
    return c


# ----------------------------------------------------------------------------
# Curriculum state
# ----------------------------------------------------------------------------

class Curriculum:
    def __init__(self, n: int, seed: int):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.difficulty = np.zeros(n, dtype=np.float32)
        self.uncertainty = np.ones(n, dtype=np.float32)
        self.miss = np.zeros(n, dtype=np.float32)
        self.seen = np.zeros(n, dtype=bool)
        self.order = self.rng.permutation(n)
        self.ptr = 0
        self.replay_pool: List[int] = []

    def next_random(self, bs: int) -> np.ndarray:
        if self.ptr + bs > self.n:
            self.order = self.rng.permutation(self.n)
            self.ptr = 0
        idx = self.order[self.ptr : self.ptr + bs]
        self.ptr += bs
        return idx

    def next_weighted(self, bs: int, weights: np.ndarray) -> np.ndarray:
        w = np.clip(np.asarray(weights, dtype=np.float64), 1e-8, None)
        w = w / w.sum()
        return self.rng.choice(self.n, size=bs, replace=False, p=w)

    def next_from_pool(self, bs: int, pool: List[int]) -> np.ndarray:
        if not pool:
            return self.next_random(bs)
        k = min(bs, len(pool))
        return self.rng.choice(np.asarray(pool), size=k, replace=False)

    def mark_seen(self, idx: np.ndarray):
        self.seen[idx] = True
        self.replay_pool.extend(int(i) for i in idx)
        if len(self.replay_pool) > 20000:
            self.replay_pool = self.replay_pool[-20000:]


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------

@dataclass
class RunResult:
    condition: str
    seed: int
    params: int
    test_acc: float = 0.0
    test_f1: float = 0.0
    test_nll: float = 0.0
    val_acc: float = 0.0
    val_f1: float = 0.0
    final_train_acc: float = 0.0
    grad_evals: int = 0
    probe_grad_evals: int = 0
    examples_seen: int = 0
    wall_s: float = 0.0
    peak_mem_mb: float = 0.0
    retention: float = 0.0
    forgetting: float = 0.0
    stage_log: List[Dict] = field(default_factory=list)
    generations: List[Dict] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------

class HLCTrainer:
    def __init__(
        self,
        condition: str,
        seed: int,
        data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        policy: Policy,
        batch_size: int = 128,
        threads: int = 8,
        out_dir: str = ".",
        retention_set: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_generations: int = 3,
    ):
        assert condition in ("adamw", "hlc_fixed", "hlc_adaptive", "sihlc")
        self.condition = condition
        self.seed = seed
        self.policy = policy
        self.bs = batch_size
        self.out_dir = out_dir
        self.threads = threads
        torch.set_num_threads(threads)

        # full determinism
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cpu")
        self.xtr, self.ytr = data["train"]
        self.xval, self.yval = data["val"]
        self.xte, self.yte = data["test"]

        self.model = build_model()
        self.params = count_params(self.model)
        assert self.params == TARGET_PARAMS, f"{self.params} != {TARGET_PARAMS}"
        init_weights(self.model, seed)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=policy.lr, weight_decay=1e-2)
        self.exp = Experience(miss_counts=np.zeros(len(self.xtr), dtype=np.float32))
        self.cur = Curriculum(len(self.xtr), seed + 1)

        # retention set: the easy BIRTH subset, held out of later training
        self.retention_set = retention_set
        self.stage_log: List[Dict] = []
        self.generations: List[Dict] = []
        self.max_generations = max_generations
        self.t0 = None
        self.peak_mem = 0.0

    # -- helpers ------------------------------------------------------------

    def _mem(self):
        try:
            import resource
            m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            self.peak_mem = max(self.peak_mem, m)
        except Exception:
            pass

    def _step(self, idx: np.ndarray, lr: Optional[float] = None) -> Tuple[float, float]:
        xb = self.xtr[idx]
        yb = self.ytr[idx]
        if lr is not None:
            for g in self.opt.param_groups:
                g["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        logits = self.model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            err = (logits.argmax(1) != yb).float().mean().item()
        return float(loss.item()), err

    def _update_signals(self, idx: np.ndarray):
        """Refresh difficulty / uncertainty / miss counts on the given indices."""
        xb = self.xtr[idx]
        yb = self.ytr[idx]
        with torch.no_grad():
            logits = self.model(xb)
            loss_vec = F.cross_entropy(logits, yb, reduction="none")
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        self.cur.difficulty[idx] = loss_vec.cpu().numpy()
        self.cur.uncertainty[idx] = (1.0 - conf).cpu().numpy()
        wrong = (pred != yb).cpu().numpy()
        self.cur.miss[idx] = self.cur.miss[idx] * 0.9 + wrong * 0.1
        self.cur.mark_seen(idx)

    def _retention(self) -> float:
        if self.retention_set is None:
            return float("nan")
        xr, yr = self.retention_set
        m = evaluate(self.model, xr, yr, batch_size=512)
        return m["accuracy"]

    def _log_stage(self, stage: str, step: int, note: str = ""):
        rec = {
            "stage": stage,
            "step": step,
            "grad_evals": self.exp.grad_evals,
            "examples_seen": self.exp.examples_seen,
            "train_loss": self.exp.recent_loss(),
            "train_err": self.exp.recent_err(),
            "retention": self._retention(),
            "note": note,
        }
        self.stage_log.append(rec)

    # -- the six stages ----------------------------------------------------

    def _run_stage_budget(self, fn, budget: int):
        """Run a stage function for its full budget.

        Used by the fixed-schedule condition so the step count per stage is
        deterministic and the total is compute-matched to the baseline.
        `fn` is expected to already suppress early transitions.
        """
        fn(budget)

    def stage_birth(self, budget: int, hold: bool = False) -> str:
        """Learn foundational patterns from relatively easy examples."""
        # Score difficulty once with the untrained-ish model, take the easiest slice
        d = score_difficulty(self.model, self.xtr[:4000], self.ytr[:4000])
        thr = np.quantile(d, 0.25)
        easy_mask = d <= thr
        easy_idx = np.where(easy_mask)[0]
        self.easy_pool = easy_idx.tolist()

        for s in range(budget):
            bs = self.bs
            if len(easy_idx) >= bs:
                idx = self.cur.rng.choice(easy_idx, size=bs, replace=False)
            else:
                idx = self.cur.next_random(bs)
            l, e = self._step(idx)
            self.exp.record_batch(l, e, len(idx))
            if s % 10 == 0:
                self._update_signals(idx)
            self._mem()
            # transition rule (suppressed when the schedule holds the stage)
            if (not hold and s >= self.policy.min_stage_steps
                    and self.exp.recent_err(30) <= (1.0 - self.policy.birth_to_growth_acc)):
                return "GROWTH"
        return "GROWTH"

    def stage_growth(self, budget: int, hold: bool = False) -> str:
        """Gradually increase difficulty and diversity."""
        for s in range(budget):
            n_hard = int(self.bs * self.policy.p_hard)
            n_rand = self.bs - n_hard
            w = self.cur.difficulty + 1e-6
            hard_idx = self.cur.next_weighted(n_hard, w)
            rand_idx = self.cur.next_random(n_rand)
            idx = np.concatenate([hard_idx, rand_idx])
            l, e = self._step(idx)
            self.exp.record_batch(l, e, len(idx))
            if s % 5 == 0:
                self._update_signals(idx)
            self._mem()
            if (not hold and s >= self.policy.min_stage_steps
                    and self.exp.recent_err(30) <= self.policy.growth_to_expl_err):
                return "EXPLORATION"
        return "EXPLORATION"

    def stage_exploration(self, budget: int, hold: bool = False) -> str:
        """Prioritise uncertain examples and unfamiliar patterns."""
        for s in range(budget):
            n_unc = int(self.bs * self.policy.p_uncertain)
            n_rand = self.bs - n_unc
            w = self.cur.uncertainty + 1e-6
            unc_idx = self.cur.next_weighted(n_unc, w)
            rand_idx = self.cur.next_random(n_rand)
            idx = np.concatenate([unc_idx, rand_idx])
            l, e = self._step(idx)
            self.exp.record_batch(l, e, len(idx))
            if s % 5 == 0:
                self._update_signals(idx)
            self._mem()
            if (not hold and s >= self.policy.min_stage_steps
                    and self.exp.learning_progress(50) < self.policy.expl_to_refl_progress):
                return "REFLECTION"
        return "REFLECTION"

    def stage_reflection(self, budget: int, hold: bool = False) -> str:
        """Revisit examples where the model repeatedly makes mistakes."""
        for s in range(budget):
            n_miss = int(self.bs * self.policy.p_missed)
            n_rand = self.bs - n_miss
            w = self.cur.miss + 1e-6
            miss_idx = self.cur.next_weighted(n_miss, w)
            rand_idx = self.cur.next_random(n_rand)
            idx = np.concatenate([miss_idx, rand_idx])
            l, e = self._step(idx)
            self.exp.record_batch(l, e, len(idx))
            if s % 5 == 0:
                self._update_signals(idx)
            self._mem()
            if (not hold and s >= self.policy.min_stage_steps
                    and self._retention() >= self.policy.refl_to_cons_retention):
                return "CONSOLIDATION"
        return "CONSOLIDATION"

    def stage_consolidation(self, budget: int, hold: bool = False) -> str:
        """Mix previous examples with new material to reduce forgetting."""
        for s in range(budget):
            n_old = int(self.bs * self.policy.p_replay)
            n_new = self.bs - n_old
            old_idx = self.cur.next_from_pool(n_old, self.cur.replay_pool)
            new_idx = self.cur.next_random(n_new)
            idx = np.concatenate([old_idx, new_idx])
            l, e = self._step(idx)
            self.exp.record_batch(l, e, len(idx))
            if s % 5 == 0:
                self._update_signals(idx)
            self._mem()
            if (not hold and s >= self.policy.min_stage_steps
                    and self.exp.stagnation(100) < self.policy.cons_to_renew_stag):
                return "RENEWAL"
        return "RENEWAL"

    def stage_renewal(self, budget: int, generation: int) -> Optional[Policy]:
        """
        Evaluate learning progress, retention, stagnation; then (sihlc only)
        run a generational policy-selection tournament from this checkpoint.
        Returns the policy to continue with (incumbent if no candidate wins).
        """
        self._log_stage("RENEWAL", self.exp.grad_evals,
                        f"gen={generation} stagnation={self.exp.stagnation(100):.5f}")
        if self.condition != "sihlc":
            # adaptive / fixed: no policy evolution
            return None

        # ---- generational selection -------------------------------------
        # Snapshot model + optimizer so every candidate starts identically.
        model_state = copy.deepcopy(self.model.state_dict())
        opt_state = copy.deepcopy(self.opt.state_dict())
        incumbent = self.policy.clone()
        base_val = evaluate(self.model, self.xval, self.yval, batch_size=1024)

        rng = np.random.default_rng(self.seed * 1000 + generation)
        candidates = [mutate(incumbent, rng) for _ in range(3)]

        results = []
        for ci, cand in enumerate(candidates):
            # restore identical starting point
            self.model.load_state_dict(model_state)
            self.opt.load_state_dict(opt_state)
            for g in self.opt.param_groups:
                g["lr"] = cand.lr
            # Equal, bounded probe budget for every candidate. These steps are
            # EXPLORATORY: they are counted separately (probe_grad_evals) and
            # are NOT part of the lineage's training budget, because the
            # checkpoint is restored afterwards either way.
            probe_budget = max(20, min(60, self.policy.d_consolidation // 2))
            probe_grads_before = self.exp.grad_evals
            exp_backup = copy.deepcopy(self.exp)
            self.exp.loss_window = list(self.exp.loss_window[-200:])
            self.exp.err_window = list(self.exp.err_window[-200:])
            for s in range(probe_budget):
                n_old = int(self.bs * cand.p_replay)
                old_idx = self.cur.next_from_pool(n_old, self.cur.replay_pool)
                new_idx = self.cur.next_random(self.bs - n_old)
                idx = np.concatenate([old_idx, new_idx])
                l, e = self._step(idx)
                self.exp.record_batch(l, e, len(idx))
            cand_val = evaluate(self.model, self.xval, self.yval, batch_size=1024)
            results.append({
                "candidate": ci,
                "policy": cand.to_dict(),
                "val_acc": cand_val["accuracy"],
                "val_f1": cand_val["macro_f1"],
                "probe_budget": probe_budget,
            })
            # restore experience bookkeeping; probe steps are tracked apart
            probe_used = self.exp.grad_evals - probe_grads_before
            self.exp = exp_backup
            self.exp.probe_grad_evals += probe_used

        best = max(results, key=lambda r: r["val_acc"])
        incumbent_score = base_val["accuracy"]
        accepted = best["val_acc"] > incumbent_score

        gen_rec = {
            "generation": generation,
            "step": self.exp.grad_evals,
            "incumbent_val_acc": incumbent_score,
            "candidates": results,
            "accepted": accepted,
            "selected": "candidate" if accepted else "incumbent",
            "selected_val_acc": best["val_acc"] if accepted else incumbent_score,
        }
        self.generations.append(gen_rec)

        # restore the pre-tournament checkpoint so the lineage is never
        # advanced by a rejected experiment
        self.model.load_state_dict(model_state)
        self.opt.load_state_dict(opt_state)
        if accepted:
            return best["policy"] and Policy.from_dict(best["policy"])
        return incumbent.clone()

    # -- main loops --------------------------------------------------------

    def run_fixed(self, total_budget: int):
        """Condition B: transitions purely on step counts.

        Stage budgets are scaled so the total equals `total_budget`,
        making B compute-matched to condition A.
        """
        raw = [
            self.policy.d_birth,
            self.policy.d_growth,
            self.policy.d_exploration,
            self.policy.d_reflection,
            self.policy.d_consolidation,
        ]
        # RENEWAL performs no gradient updates (it is evaluation + optional
        # policy selection), so the whole budget is split across the five
        # active stages and the total gradient count equals total_budget.
        tot = sum(raw)
        plan_steps = [max(1, round(b * total_budget / tot)) for b in raw]
        # fix rounding drift so the sum is exactly total_budget
        drift = total_budget - sum(plan_steps)
        plan_steps[0] += drift
        plan = list(zip(STAGES[:5], plan_steps))
        fns = {
            "BIRTH": lambda b: self.stage_birth(b, hold=True),
            "GROWTH": lambda b: self.stage_growth(b, hold=True),
            "EXPLORATION": lambda b: self.stage_exploration(b, hold=True),
            "REFLECTION": lambda b: self.stage_reflection(b, hold=True),
            "CONSOLIDATION": lambda b: self.stage_consolidation(b, hold=True),
        }
        for stage, budget in plan:
            self._log_stage(stage, self.exp.grad_evals,
                            f"fixed schedule budget={budget}")
            if stage in fns:
                # run the whole stage budget regardless of early-transition
                # signals, so the step count stays deterministic
                self._run_stage_budget(fns[stage], budget)

    def run_adaptive(self, total_budget: int):
        """
        Conditions C and D: transitions driven by experience signals.

        The six stages run as a cycle. Each stage gets a planned share of the
        generation budget, but a stage that satisfies its transition rule early
        does NOT abandon the remaining steps: the surplus is carried forward to
        the next stage in the cycle. Total gradient evaluations are therefore
        exactly `total_budget`, keeping C and D compute-matched to A and B.
        """
        # Condition C (hlc_adaptive) runs a SINGLE life-cycle pass, so its one
        # generation must absorb the entire budget. Condition D (sihlc) repeats
        # the cycle for max_generations generations.
        n_gens = 1 if self.condition == "hlc_adaptive" else self.max_generations
        per_gen = max(1, total_budget // n_gens)
        for gen in range(n_gens):
            remaining = total_budget - self.exp.grad_evals
            if remaining <= 0:
                break
            gen_budget = min(per_gen, remaining)
            gen_start = self.exp.grad_evals
            # planned share of the generation budget per stage
            weights = [
                self.policy.d_birth,
                self.policy.d_growth,
                self.policy.d_exploration,
                self.policy.d_reflection,
                self.policy.d_consolidation,
            ]
            tot = sum(weights)
            planned = [max(1, round(w * gen_budget / tot)) for w in weights]
            drift = gen_budget - sum(planned)
            planned[0] += drift

            stage_fns = [
                ("BIRTH", self.stage_birth),
                ("GROWTH", self.stage_growth),
                ("EXPLORATION", self.stage_exploration),
                ("REFLECTION", self.stage_reflection),
                ("CONSOLIDATION", self.stage_consolidation),
            ]

            carried = 0
            for si, (name, fn) in enumerate(stage_fns):
                want = planned[si] + carried
                before = self.exp.grad_evals
                self._log_stage(name, before, f"gen={gen} want={want}")
                fn(want)                       # may return early
                used = self.exp.grad_evals - before
                carried = max(0, want - used)   # surplus rolls to next stage

            # Any surplus left after the last active stage must still be spent,
            # otherwise C and D would under-use their compute budget. Spend it
            # in CONSOLIDATION (replay + new material), which is the stage whose
            # job is to keep learning without destabilising the model.
            guard = 0
            while carried > 0 and guard < 10:
                before = self.exp.grad_evals
                self.stage_consolidation(carried)
                used = self.exp.grad_evals - before
                if used <= 0:
                    break
                carried = max(0, carried - used)
                guard += 1

            # hard top-up: if the generation still came up short, run plain
            # consolidation steps so the compute budget is exactly matched
            short = gen_budget - (self.exp.grad_evals - gen_start)
            if short > 0:
                self.stage_consolidation(short, hold=True)

            new_policy = self.stage_renewal(0, gen)
            if new_policy is not None:
                self.policy = new_policy
            if self.condition == "hlc_adaptive":
                break

    def run(self, total_budget: int = 3000):
        self.t0 = time.time()
        if self.condition == "adamw":
            self.run_adamw(total_budget)
        elif self.condition == "hlc_fixed":
            self.run_fixed(total_budget)
        else:
            self.run_adaptive(total_budget)
        wall = time.time() - self.t0
        self._mem()

        val = evaluate(self.model, self.xval, self.yval, batch_size=1024)
        test = evaluate(self.model, self.xte, self.yte, batch_size=1024)
        tr = evaluate(self.model, self.xtr[:10000], self.ytr[:10000], batch_size=1024)
        ret = self._retention()
        # forgetting = 1 - retention
        forg = 1.0 - ret if ret == ret else float("nan")

        return RunResult(
            condition=self.condition,
            seed=self.seed,
            params=self.params,
            test_acc=test["accuracy"],
            test_f1=test["macro_f1"],
            test_nll=test["nll"],
            val_acc=val["accuracy"],
            val_f1=val["macro_f1"],
            final_train_acc=tr["accuracy"],
            grad_evals=self.exp.grad_evals,
            probe_grad_evals=self.exp.probe_grad_evals,
            examples_seen=self.exp.examples_seen,
            wall_s=wall,
            peak_mem_mb=self.peak_mem,
            retention=ret,
            forgetting=forg,
            stage_log=self.stage_log,
            generations=self.generations,
        )

    def run_adamw(self, total_budget: int):
        """Condition A: plain AdamW, uniform shuffled minibatches."""
        self._log_stage("ADAMW", 0, "baseline")
        for s in range(total_budget):
            idx = self.cur.next_random(self.bs)
            l, e = self._step(idx)
            self.exp.record_batch(l, e, len(idx))
            if s % 20 == 0:
                self._update_signals(idx)
            self._mem()
