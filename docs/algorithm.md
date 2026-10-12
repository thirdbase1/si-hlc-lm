# The shared algorithm

Both experiments in this repo run the same SI-HLC training loop. This file documents the
mechanisms that are common to both, so neither subdirectory has to re-derive them.

## Signals

For each training unit (an image in MNIST, a packed 256-token block in the LM) the trainer keeps:

| Signal | Definition | Used by |
|---|---|---|
| `difficulty` | mean cross-entropy of the unit under the current model | BIRTH, GROWTH |
| `uncertainty` | mean predictive entropy of the unit | EXPLORATION |
| `miss` | EMA of token/unit-level error | REFLECTION |
| `replay` | indices of units already trained on | CONSOLIDATION |

Signals are refreshed over a bounded sample (512 units), **chunked over 32 at a time**. This is not
an optimisation detail: a 512-block forward pass over a 7,618-token vocabulary materialises
`(512, 256, 7618)` = 4 GB of logits, and the softmax and reduction add another 4–7 GB of
transients. Peak RSS reached 20 GB while steady-state RSS stayed at 0.6 GB, so `ps` looked
healthy and only `VmHWM` in `/proc/<pid>/status` exposed it.

## Stages

| Stage | Sampling bias | Transition rule |
|---|---|---|
| BIRTH | easiest 25 % | recent loss ≤ `birth_to_growth_loss` **and** `step ≥ min_stage_steps` |
| GROWTH | anneals easy → difficulty-weighted | recent loss ≤ `growth_to_expl_loss` **and** `step ≥ min_stage_steps` |
| EXPLORATION | 50 % entropy-weighted + uniform | learning progress < `progress_epsilon` **and** `step ≥ min_stage_steps` |
| REFLECTION | 70 % miss-weighted + uniform | retention loss ≤ `refl_to_cons_retention` **and** `step ≥ min_stage_steps` |
| CONSOLIDATION | 30 % replay + 70 % new | stagnation < `cons_to_renew_stagnation` **and** `step ≥ min_stage_steps` |
| RENEWAL | evaluation only | logs progress/retention/stagnation, then runs the tournament |

`learning_progress` = mean loss over the last *k* steps minus the previous *k*.
`stagnation` = relative absolute change.

Every rule has a `min_stage_steps` floor. Without it a threshold satisfied by an untrained model
fires at step 1 and the stage never runs — this was a real bug, not a hypothetical one.

## Thresholds do not transfer between tasks

The thresholds are scale-dependent and must be re-derived per task, never copied:

- MNIST: a loss lives in 0–1; a trained MLP sits near 0.05.
- LM: an untrained model sits at `ln(vocab)` (8.94 for V=7,618) and a trained one near 4.5–6.9.

Copying MNIST's 0–1 thresholds into the LM trainer produced a "diverging" model that was in fact
just badly warmupped. Also make `warmup_steps` a **fraction** of the total budget, so a 60-step
smoke run does not spend its entire budget inside warmup.

## Generational policy tournament (RENEWAL)

1. Snapshot `model.state_dict()` **and** `optimizer.state_dict()`.
2. Generate `n_candidates` mutants from the incumbent `Policy` (`lr`, `warmup_frac`, stage
   durations, sampling proportions, thresholds — all clamped to sane ranges).
3. Restore each candidate to the identical checkpoint and give it an **equal probe budget**
   (identical for every candidate; an unequal budget is an unfair comparison).
4. Score on **validation** loss. A candidate replaces the incumbent only if it is **strictly**
   better.
5. Restore the pre-tournament checkpoint either way, so a rejected experiment never advances the
   lineage.

Probe steps are tracked as `probe_steps` / `probe_tokens` and **excluded** from the lineage budget:
the step and token counters are rolled back (`Experience.unrecord_step`) after each tournament.
This is why the generational condition costs ~77 % more wall-clock than the baseline — the probes
are real compute, just not lineage compute.

## Compute matching

Every condition performs an identical number of optimiser steps and sees an identical number of
tokens. This is asserted in the driver, not assumed. Three failure modes that silently break it:

- **A stage that returns early** without spending its remaining steps under-trains that condition.
  Fix: carry surplus steps forward.
- **An evaluation-only stage** included in the budget split makes the condition run short.
- **Probe steps folded into the lineage budget** inflate one condition's compute relative to the
  others.

## Reporting

Report per-seed values, mean ± std, and **paired** per-seed differences against the baseline. A
mean gap that is small relative to the seed spread is not a result — on MNIST, +0.0011 accuracy is
11 images out of 10,000.

And the rule this repo exists to enforce: **a loss win is not a capability win.** When a condition
beats the baseline on loss or perplexity, still run the task-level metrics (QA token-F1, exact
match, memorisation, multi-turn continuity) and report them with equal prominence. The LM run
shows a 41 % perplexity gain coexisting with 0.000 exact match and semantically empty output —
high distinct-n with zero 3-gram repetition proves local syntax, nothing more.
