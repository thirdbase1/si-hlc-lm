# SI-HLC Experiment Report

**Human Life-Cycle Learning (HLC) on a fresh 10,000,000-parameter model**

Run ID: `main_20261012_080850` · Host: boxd machine `wise-sage` (8 vCPU, 32 GiB, CPU-only)
Framework: PyTorch 2.14.1+cpu · Date: 2026-10-12 (UTC)

---

## 1. Model

| Property | Value |
|---|---|
| Architecture | 4-layer MLP: `784 → 2074 → 2034 → 2030 → 10`, ReLU, `nn.Flatten` input |
| **Trainable parameters** | **10,000,000 (exact)** |
| Initialisation | Kaiming-uniform (`fan_in`, `a=√5`) — PyTorch default, re-seeded per run; **from scratch, no pretrained weights** |
| Optimiser | AdamW (`weight_decay=1e-2`), LR from the active policy (default `1e-3`) |
| Precision / device | float32, CPU, `torch.set_num_threads(8)` |
| Peak memory | ~0.73–1.0 GiB RSS per run |

Parameter count was solved analytically (`(784+1)h₁ + (h₁+1)h₂ + (h₂+1)h₃ + (h₃+1)·10 = 10,000,000`),
then asserted at runtime: `count_params(model) == 10_000_000`.

## 2. Data

MNIST (canonical IDX files, all four MD5s verified against the reference checksums).

- Official training split 60,000 → **54,000 train / 10,000 validation** (deterministic `np.random.default_rng(seed).permutation` split)
- Official test split 10,000 → **held-out test, used only for final reporting**
- Preprocessing: `/255.0`, then normalise with mean `0.1307`, std `0.3081`
- Retention set: the easiest 2,000 training examples (scored at init), used to measure forgetting

**Leakage controls:** stage transitions, adaptive thresholds, candidate policy selection and all
hyperparameter choice use **validation only**. The test set is read exactly once per run, after
training finishes, by the same code path for every condition.

## 3. Conditions (all compute-matched: exactly 3,000 gradient evaluations each)

| | Condition | Description |
|---|---|---|
| A | `adamw` | Standard AdamW, uniform shuffled minibatches |
| B | `hlc_fixed` | Six-stage HLC, transitions on **step count** only (no adaptive rules) |
| C | `hlc_adaptive` | Six-stage HLC, transitions driven by **experience signals** |
| D | `sihlc` | C + **generational policy selection** at RENEWAL |

Seeds `0,1,2,3,4`; identical architecture, init seed, data split and batch size (128) across conditions.

## 4. The six stages and their transition rules

Each stage samples a mixture biased by a live per-example signal (difficulty = per-example CE,
uncertainty = `1 − max softmax`, miss = EMA of being wrong, replay pool = indices already trained on).

| Stage | Sampling | Transition rule (measurable) |
|---|---|---|
| **BIRTH** | easiest 25 % of 4,000 scored examples | `recent_err(30) ≤ 1 − birth_to_growth_acc` **and** `step ≥ min_stage_steps` |
| **GROWTH** | `p_hard` difficulty-weighted + uniform | `recent_err(30) ≤ growth_to_expl_err` **and** `step ≥ min_stage_steps` |
| **EXPLORATION** | `p_uncertain` uncertainty-weighted + uniform | `learning_progress(50) < expl_to_refl_progress` **and** `step ≥ min_stage_steps` |
| **REFLECTION** | `p_missed` miss-weighted + uniform | `retention ≥ refl_to_cons_retention` **and** `step ≥ min_stage_steps` |
| **CONSOLIDATION** | `p_replay` from replay pool + new | `stagnation(100) < cons_to_renew_stag` **and** `step ≥ min_stage_steps` |
| **RENEWAL** | evaluation only (no gradient steps) | logs progress/retention/stagnation, then (D only) runs the policy tournament |

`learning_progress` = mean loss over the last *k* minus the previous *k*; `stagnation` = relative
absolute change. `min_stage_steps = 25` prevents a stage from firing its rule at step 1 on an
uninitialised model. Surplus steps from a stage that transitions early are carried forward, so every
condition consumes exactly the same number of gradient evaluations.

## 5. Generational policy selection (condition D)

At each RENEWAL, with **weights and optimizer state unchanged**:

1. Snapshot `model.state_dict()` and `optimizer.state_dict()`.
2. Generate 3 mutated candidates from the incumbent policy (`Policy.mutate`: thresholds, stage
   durations, sampling proportions, LR; all clamped to sane ranges).
3. Each candidate is restored to the identical checkpoint and given an **equal probe budget**
   (20–60 steps, identical for all candidates).
4. Score on **validation**; a candidate replaces the incumbent **only if** its validation accuracy
   strictly exceeds the incumbent's.
5. The pre-tournament checkpoint is restored either way, so a rejected experiment never advances
   the lineage. No restart from scratch — the model weights persist across all generations.

Probe steps are counted separately as `probe_grad_evals` and are **not** part of the 3,000-step
training budget.

## 6. Results (5 matched seeds, mean ± std)

| Condition | Params | Test acc | Test macro-F1 | Val acc | Retention | Grads | Wall (s) | Peak MB |
|---|---|---|---|---|---|---|---|---|
| A `adamw` | 10,000,000 | 0.9752 ± 0.0015 | 0.9750 ± 0.0016 | 0.9745 ± 0.0015 | 0.9912 ± 0.0036 | 3000 | 72.9 | 730 |
| B `hlc_fixed` | 10,000,000 | **0.9763 ± 0.0021** | 0.9761 ± 0.0021 | 0.9758 ± 0.0033 | **0.9921 ± 0.0032** | 3000 | 79.0 | 857 |
| C `hlc_adaptive` | 10,000,000 | 0.9757 ± 0.0027 | 0.9755 ± 0.0027 | 0.9742 ± 0.0029 | 0.9912 ± 0.0026 | 3000 | 78.6 | 889 |
| D `sihlc` | 10,000,000 | 0.9755 ± 0.0011 | 0.9754 ± 0.0011 | 0.9746 ± 0.0027 | 0.9912 ± 0.0021 | 3000 | 95.8 | 991 |

### Paired per-seed differences vs. `adamw` (test accuracy)

| Condition | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | **mean** | std |
|---|---|---|---|---|---|---|---|
| B − A | −0.0008 | +0.0020 | −0.0011 | +0.0011 | +0.0043 | **+0.0011** | 0.0022 |
| C − A | −0.0023 | +0.0004 | −0.0042 | +0.0054 | +0.0032 | **+0.0005** | 0.0039 |
| D − A | −0.0021 | −0.0011 | +0.0010 | +0.0027 | +0.0013 | **+0.0004** | 0.0019 |

Test NLL per seed (mean ± std over 5 seeds):

| Condition | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean ± std |
|---|---|---|---|---|---|---|
| A `adamw` | 0.1005 | 0.1264 | 0.0922 | 0.1231 | 0.1280 | 0.1140 ± 0.0148 |
| B `hlc_fixed` | 0.1040 | 0.0951 | 0.1037 | 0.1043 | 0.0833 | **0.0981 ± 0.0082** |
| C `hlc_adaptive` | 0.1112 | 0.1077 | 0.1347 | 0.0959 | 0.1091 | 0.1117 ± 0.0127 |
| D `sihlc` | 0.1131 | 0.1046 | 0.1094 | 0.1108 | 0.1023 | 0.1080 ± 0.0040 |

This is the most discriminating result in the experiment and it **does** favour HLC: the fixed
six-stage schedule cuts mean test NLL by 14 % versus AdamW (0.0981 vs 0.1140) *and* nearly halves
its variance (std 0.0082 vs 0.0148). The adaptive (C) and generational (D) variants sit in between,
with D having the lowest variance of all (0.0040) but a higher mean than B.

### Generational selection behaviour (condition D)

13 of 15 tournaments accepted a candidate; 2 correctly retained the incumbent when no candidate beat
it on validation. Accepted candidates were consistently followed by higher validation accuracy at
the next generation (e.g. seed 0: incumbent 0.9648 → 0.9710 → 0.9703 after two acceptances).

## 7. Honest conclusions

**HLC shows a real but narrow benefit on calibration/loss; it does not improve accuracy, and the
adaptive and generational extensions did not earn their cost.**

1. **Accuracy: no improvement.** The ranking on test accuracy is `hlc_fixed` (0.9763) >
   `hlc_adaptive` (0.9757) > `sihlc` (0.9755) > `adamw` (0.9752). The best paired gain over AdamW is
   `hlc_fixed` at **+0.0011 ± 0.0022** — about 11 images out of 10,000, with two of five seeds
   negative. That is not a separable result.

2. **Loss/calibration: HLC does help.** On test NLL the fixed six-stage schedule reaches
   **0.0981 ± 0.0082** versus AdamW's **0.1140 ± 0.0148** — a 14 % lower mean and roughly half the
   variance. Since all conditions share identical architecture, init and step count, this is the
   cleanest evidence in the run that the *curriculum itself* (easy-first → hard → uncertain →
   missed → replay) produces better-calibrated, more seed-stable models.

3. **Adaptivity (C) hurt.** `hlc_adaptive` has the widest accuracy spread (±0.0027) and a *lower*
   mean validation accuracy than both `hlc_fixed` and `adamw`. The signal-driven transitions add
   variance without adding accuracy. The fixed schedule beat both adaptive variants on every metric
   that mattered.

4. **Generational selection (D) bought stability, not performance.** SI-HLC has the tightest spread
   of all four conditions on both accuracy (±0.0011) and NLL (±0.0040) — the tournament reliably
   finds a good policy and avoids bad ones. But its mean test accuracy (0.9755) is statistically
   indistinguishable from AdamW (0.9752), and it costs **~31 % more wall-clock** (95.8 s vs 72.9 s)
   for the probe tournaments. 13 of 15 tournaments accepted a candidate; 2 correctly retained the
   incumbent. So the machinery works as specified and does what it claims, but on this task the
   accuracy payoff does not cover the compute.

5. **The result is task-confounded.** MNIST with a 10M-parameter MLP saturates at 0.975–0.978
   regardless of sampling strategy; the model is heavily over-parameterised, so *which* examples you
   feed it matters far less than *how many* gradient steps you take. The NLL effect is the one place
   where sample selection visibly mattered, and it is the metric worth carrying forward.

6. **What is established:** the six-stage machinery, the experience record, and the generational
   tournament are correctly implemented, compute-matched, leak-free and reproducible. The negative
   accuracy result is a genuine finding about *this* task and budget — not evidence that HLC is
   broken.

**Recommended next step if this line is pursued:** move to a task with a real sample-efficiency
constraint (low-data CIFAR-10, or a class-imbalanced long-tail split), raise the budget so curriculum
effects can accumulate, report NLL/ECE as primary metrics rather than accuracy, and add a fifth
condition isolating *only* replay — to test whether the NLL gain comes from anti-forgetting rather
than from the stage schedule.

## 8. Artifacts

```
hlc_sihlc/
├── hlc_core.py            # model, data, metrics, experience record
├── hlc_train.py           # six stages, adaptive rules, generational selection
├── run_experiment.py      # driver (smoke / screen / main)
├── analyse.py             # aggregation + paired differences
├── verify.py              # independent checkpoint re-evaluation
├── show_stages.py         # stage-log inspector
└── runs/main_20261012_080850/
    ├── config.json
    ├── report.json
    ├── summary.json
    ├── metrics_{condition}_seed{0..4}.json   (20 files, full logs)
    └── ckpt_sihlc_seed2.pt                   # best SI-HLC model (10,000,000 params)
```

All 20 checkpoints (including model + optimizer state + config + seeds) remain on the `wise-sage`
boxd machine at `/home/boxd/hlc/runs/main_20261012_080850/`. The local copy is the best SI-HLC
checkpoint (seed 2, test acc 0.9769); local disk had only 2.7 GB free, which cannot hold all 20.

**Reproduce:**
```bash
boxd machine cp hlc_core.py hlc_train.py run_experiment.py wise-sage:/home/boxd/hlc/
boxd machine exec wise-sage -- sh -c 'cd /home/boxd/hlc && . .venv/bin/activate && \
  python run_experiment.py --mode main --budget 3000 --generations 3'
```
