# SI-HLC

Experiments testing whether a **human-life-cycle-inspired (HLC)** training algorithm improves
learning in small neural networks, with matched-compute controls and honest reporting.

Two independent experiments share one training algorithm but **not** their conclusions:

| | [MNIST MLP](mnist/) | [Language model](lm/) |
|---|---|---|
| Task | digit classification | next-token prediction on English dialogue |
| Architecture | 784→2074→2034→2030→10 MLP | decoder-only Transformer, `d=320 L=6 ffn=1280` |
| Params | 10,000,000 (exact) | 10,000,000 (exact) |
| Dataset | MNIST | DailyDialog (CC-BY-NC-SA-4.0, non-commercial) |
| Run | `main_20261012_080850` | `main_20261012_113306` |
| Seeds × conditions | 5 × 4 = 20 runs | 5 × 4 = 20 runs |
| **Did HLC help?** | **Narrowly, on calibration only** | **Yes on held-out loss (5/5 seeds), no on capability** |

Both parameter counts are asserted at runtime and cross-checked against a closed form; in both
cases the tokenizer/hidden sizes were *derived from* the parameter budget rather than the other
way round.

## The algorithm

Six stages, each with an explicit measurable transition rule:

```
BIRTH → GROWTH → EXPLORATION → REFLECTION → CONSOLIDATION → RENEWAL (× generations)
```

Sampling is biased by live per-example signals — difficulty (mean loss), uncertainty (mean
predictive entropy), miss (EMA of error), replay pool (already-trained indices). RENEWAL runs a
**generational policy tournament**: snapshot weights *and* optimizer state, probe mutated
candidate policies on an equal budget, score on **validation** loss, keep a candidate only if it
strictly wins, then restore the pre-tournament checkpoint regardless. Weights persist across
generations — nothing restarts from scratch.

Four conditions are compared under matched compute: `adamw` (baseline), `hlc_fixed`,
`hlc_adaptive`, and `sihlc` (adaptive + generational selection).

## Findings in one line each

- **MNIST**: `hlc_fixed` cut test NLL 14 % (0.0981 vs 0.1140) and halved its variance, but did
  **not** improve accuracy (best paired gain +0.0011 ± 0.0022 — 11 images out of 10,000, inside
  noise). Adaptivity added variance; generational selection bought stability, not performance,
  at ~31 % more wall-clock. The task is confounded: a 10 M-param MLP on MNIST is over-parameterised,
  so sample selection matters less than step count.
- **LM**: `sihlc` beat AdamW on held-out test loss in **all 5 seeds** (−0.6013 mean, −41 %
  perplexity), while `hlc_fixed` and `hlc_adaptive` were slightly *worse* than baseline — so the
  gain came from generational selection, not the curriculum. **But** the best model still scores
  QA token-F1 **0.089** and exact match **0.000**, and produces well-formed English with no
  semantic content. It **cannot converse**.

The two runs together are the actual result: HLC's benefit is real but task-dependent, and a loss
improvement is not a capability improvement.

## Repo layout

```
lm/          language-model experiment: code, configs, 20 metrics files, report, tokenizer
mnist/       MNIST experiment: code, 22 metrics files, report
docs/        shared notes on the algorithm and the compute-matching controls
```

## Checkpoints

All 40 checkpoints (20 LM + 20 MNIST, ~120 MB each, ~4.8 GB total) are published as assets on the
[latest GitHub release](../../releases/latest) rather than committed, because they far exceed any
sane git history. Each stores `model_state`, `opt_state`, `config`, `policy`, `seed`, and `result`.

Every checkpoint was verified independently: strict `state_dict` load with zero missing or
unexpected keys, and an independent parameter count of exactly **10,000,000**.

## Reproducing

Each subdirectory has its own README with exact commands. General shape:

```bash
# LM (≈2.6 h on 8 vCPU, CPU-only)
cd lm
python prepare_lm_data.py
OMP_NUM_THREADS=8 python run_lm_experiment.py --mode main --threads 8
python analyse_lm.py lmruns/main_<timestamp>
python evaluate_lm.py --ckpt lmruns/main_<timestamp>/ckpt_sihlc_seed1.pt --out eval.json
python chat_lm.py --ckpt lmruns/main_<timestamp>/ckpt_sihlc_seed1.pt --interactive

# MNIST
cd mnist
python run_experiment.py --mode main --budget 3000 --generations 3
python analyse.py runs/main_<timestamp>
python verify.py --ckpt runs/main_<timestamp>/ckpt_sihlc_seed2.pt
```

Environment for both: Python 3.12, `torch` (CPU build), `numpy`; the LM also needs `pyarrow` and
`tokenizers`.

## Leakage and honesty controls

- The tokenizer is trained on the **training split only** (LM experiment).
- All stage transitions, adaptive thresholds and policy selection use **validation** data.
- The **test** set is read exactly once per run, by identical code paths, after training.
- Compute is matched and asserted: every condition logs its step and token counts, and the driver
  asserts equality. Tournament probe steps are tracked separately and excluded from the lineage
  budget, with the pre-tournament checkpoint restored afterwards.
- No result in this repo is a description of an intended run — every number comes from an executed
  run, and the reports state plainly where a metric win does not imply a capability win.

## Licence

Code: MIT (see `LICENSE`). The **DailyDialog dataset is CC-BY-NC-SA-4.0 — non-commercial research
use only**; MNIST is BSD-3-Clause. Do not treat the LM model weights as cleared for commercial use.
