# SI-HLC Language Model Experiment Report

**Does a human-life-cycle-inspired curriculum improve a small causal LM's language learning?**

Run ID: `main_20261012_113306` · Host: boxd machine `wise-sage` (8 vCPU / 32 GiB, **CPU-only**)
Framework: PyTorch 2.14.1+cpu · Dataset: DailyDialog · Date: 2026-10-12 (UTC)

---

## 1. Headline result

**The generational SI-HLC condition (D) beat AdamW on held-out loss in all 5 seeds, and the
effect is large and consistent. The model, however, does not converse.**

| Condition | Test loss ↓ | Test ppl | Val ppl | Paired Δ vs AdamW | Win rate |
|---|---|---|---|---|---|
| A `adamw` | 6.9729 ± 0.0767 | 1069.9 | 992.1 | — | — |
| B `hlc_fixed` | 7.0271 ± 0.0773 | 1129.5 | 1044.2 | **+0.0542** (worse) | 0/5 |
| C `hlc_adaptive` | 7.0270 ± 0.0773 | 1129.3 | 1044.2 | **+0.0541** (worse) | 0/5 |
| D `sihlc` | **6.3716 ± 0.4263** | **627.6** | **589.4** | **−0.6013** (better) | **5/5** |

Paired per-seed test-loss differences vs AdamW:

| Condition | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std |
|---|---|---|---|---|---|---|---|
| B − A | +0.0580 | +0.0090 | +0.0711 | +0.1073 | +0.0257 | +0.0542 | 0.0387 |
| C − A | +0.0580 | +0.0088 | +0.0710 | +0.1072 | +0.0253 | +0.0541 | 0.0388 |
| D − A | **−0.2553** | **−1.1339** | **−0.1839** | **−1.0056** | **−0.4279** | **−0.6013** | 0.4391 |

Every D−A difference is negative; no B or C difference is. On perplexity that is a **41 % lower
mean test perplexity** (627.6 vs 1069.9).

**But** the best model still cannot answer a question. QA token-F1 = **0.089**, exact match
**0.000**, memorisation token-F1 = **0.089**. Its generations are well-formed English with no
semantic content (see §6).

---

## 2. Model

| Property | Value |
|---|---|
| Architecture | Decoder-only causal Transformer, **pre-norm** (GPT-2 style) |
| Config | `vocab=7618, d_model=320, n_layers=6, ffn=1280 (4×), n_heads=5, pos_max=512, tied embeddings` |
| **Trainable parameters** | **10,000,000 (exact)** |
| Initialisation | PyTorch default (`nn.Linear` kaiming-uniform), re-seeded per run; **from scratch, no pretrained weights** |
| Attention | `F.scaled_dot_product_attention(..., is_causal=True)` |
| Optimiser | AdamW (`weight_decay=0.01`, `betas=(0.9,0.95)`), linear warmup 5 % → cosine decay to 0 |
| Sequence length | 256 tokens (packed, dense, no padding) |
| Batch | 16 blocks × 256 tokens = 4,096 tokens/step |
| Precision / device | float32, CPU, 8 threads |
| Peak RSS | 1.67–1.82 GB |

**How the exact 10M was solved.** With tied embeddings the count is
`V·d + P·d + L·(4d² + 4d + 2·d·ffn) + 2d`. Fixing `d=320`, `ffn=1280`, `P=512`, the per-block cost is
1,232,960 and the residual term is 164,480, so `V = (10,000,000 − L·1,232,960 − 164,480) / 320`.
`L=6` gives **V=7618** exactly. The count is asserted at runtime
(`count_params(model) == 10_000_000`) and cross-checked against the closed form. The tokenizer
vocab size was *derived from* this constraint, not the other way round.

---

## 3. Data

**DailyDialog** (Li et al., IJCNLP 2017) — human-written multi-turn English dialogue.

- **Source**: original `ijcnlp_dailydialog` release (the author's host is dead, returns HTML);
  obtained from the HuggingFace parquet mirror `OpenRL/daily_dialog`.
- **Licence**: **CC-BY-NC-SA-4.0 — non-commercial research use only.** This is a real restriction
  and the dataset must not be used in a commercial product.
- **Splits**: 11,118 train / 1,000 validation / 1,000 test dialogues (the dataset's own splits).
  ~5.4M train characters, mean 7.8 turns and 485 characters per dialogue.
- **Preprocessing**: each dialogue rendered as an alternating `User:` / `Bot:` transcript,
  whitespace normalised, dialogues joined with a `<|doc|>` separator token, then the stream is
  chunked into **fixed 256-token blocks**.
- **Packed blocks**: train 6,109 (1,563,904 tokens), val 563 (144,128), test 551 (141,056).
  174 / 213 / 17 trailing tokens dropped. MD5s recorded in `dataset_manifest.json`.
- **Tokenizer**: byte-level BPE (HuggingFace `tokenizers`), **trained on the training split only**
  so no validation or test text can leak into the vocabulary. Vocab 7,618. Round-trip verified
  exact.

**Leakage controls.** The tokenizer sees train only. Stage transitions, adaptive thresholds and
all policy selection use the **validation** blocks. The **test** blocks are read exactly once per
run, by the same code path for every condition, after training finishes. The retention subset is
the 200 easiest *train* blocks (scored by an identically-seeded model), never val or test.

---

## 4. The six stages and their transition rules

Each stage biases minibatch sampling by a live per-example signal maintained over blocks:
difficulty = per-block mean CE, uncertainty = per-block mean predictive entropy, miss = EMA of
token-level error, replay pool = block indices already trained on.

| Stage | Sampling | Transition rule (measurable) |
|---|---|---|
| **BIRTH** | easiest 25 % of a 512-block scored sample | `recent_loss(30) ≤ 6.5` **and** `step ≥ min_stage_steps` |
| **GROWTH** | anneals from 30 % easy blocks → 100 % difficulty-weighted | `recent_loss(30) ≤ 5.0` **and** `step ≥ min_stage_steps` |
| **EXPLORATION** | 50 % entropy-weighted + uniform | `learning_progress(30) < 0.01` **and** `step ≥ min_stage_steps` |
| **REFLECTION** | 70 % miss-weighted + uniform | `retention_loss ≤ 5.0` (checked every 10 steps) **and** `step ≥ min_stage_steps` |
| **CONSOLIDATION** | 30 % replay pool + 70 % new | `stagnation(50) < 0.02` **and** `step ≥ min_stage_steps` |
| **RENEWAL** | evaluation only | logs progress/retention/stagnation, then (D only) runs the policy tournament |

`learning_progress` = mean loss over the last *k* steps minus the previous *k*; `stagnation` =
relative absolute change. Thresholds are set on the LM loss scale (untrained ≈ ln V = 8.94, trained
≈ 4.5–6.9), **not** the MNIST 0–1 scale — the thresholds were re-derived, not copied.

**These are the same mechanisms as the MNIST implementation**, re-expressed over token blocks.
The only genuinely new element is GROWTH's *annealed* difficulty mix (linear ramp from
`growth_easy_frac` to 1.0 over the stage), which replaces MNIST's fixed `p_hard` mixture because a
ramp is what "gradually increase difficulty" literally specifies.

### Compute matching

Every condition performs **exactly 600 optimiser steps × 4,096 tokens = 2,457,600 lineage
tokens**. Verified in the metrics: `steps=600` and `tokens_seen=2,457,600` for all four
conditions, all five seeds.

Condition D additionally spends **probe** steps in its tournaments (mean 253 steps, 1,037,107
tokens). These are **not** part of the lineage budget: the model and optimizer are snapshotted
before each tournament and restored afterwards, and the step/token counters are rolled back
(`Experience.unrecord_step`). Probe cost is reported separately as `probe_steps` / `probe_tokens`.
This is why D costs ~77 % more wall-clock (679 s vs 384 s) — the tournament is real compute, it is
just not lineage compute.

---

## 5. Generational policy selection (condition D)

At each RENEWAL, with **weights and optimizer state unchanged**:

1. Snapshot `model.state_dict()` and `optimizer.state_dict()`.
2. Generate 3 mutated candidates from the incumbent (`Policy.mutate`: thresholds, stage durations,
   sampling proportions, LR, warmup fraction — all clamped to sane ranges).
3. Each candidate is restored to the identical checkpoint and given an **equal probe budget**
   (20–30 steps, identical for all candidates).
4. Score on **validation loss**; a candidate replaces the incumbent **only if** its validation
   loss is strictly lower.
5. The pre-tournament checkpoint is restored either way, so a rejected experiment never advances
   the lineage. **No restart from scratch** — weights persist across all generations.

**Observed**: 10 of 15 tournaments accepted a candidate (67 %). Selection was based on validation
loss and the resulting test improvements are consistent, so the tournament is doing real work.

---

## 6. Generation quality — the honest part

Evaluated on `ckpt_sihlc_seed1.pt` (the best seed) with an independent script.

**Surface statistics look good:**

- distinct-1 = **0.976**, distinct-2 = **1.000**, 3-gram repetition rate = **0.000**

The model never repeats itself. That is a genuine achievement for 2.46 M training tokens.

**But the content is not there.** Actual generations (unedited):

```
User: Hello , how are you today ?
Bot:  I see for you .

User: What is your name ?
Bot:  I'll be for ’ s all .

User: Do you like watching movies ?
Bot:  I've .

User: Thank you , that is very helpful .
Bot:  I have a cheese .

User: Goodbye , see you later .
Bot:  It's a good .User: Can you ?
```

**Task metrics:**

| Metric | Value |
|---|---|
| QA token-F1 (n=60) | **0.0889** |
| QA exact match | **0.0000** |
| Memorisation token-F1 (n=27) | **0.0888** |
| Memorisation exact (>0.9 F1) | **0.0000** |
| Held-out test loss / ppl | 5.9145 / 370.35 |

**Multi-turn continuity fails.** The model does not reliably learn to stop at its own turn —
several "responses" continue straight into a new `User:` utterance, i.e. it has learned the
*shape* of a transcript but not the *alternation rule*:

```
[convo 2] user: We've managed to reduce our energy consumption...
   bot1: It's right . I was a job and his on the from in your name ?
   bot2: Sure .
   bot3: I've been .User: What do for a little is ?
```

### Distinguishing the three things

- **Memorised text**: essentially **none**. Near-verbatim recall of held-out training dialogues is
  0.000 at F1 > 0.9, and memorisation F1 (0.089) is statistically identical to QA F1 (0.089). The
  model is not parroting the corpus.
- **Fluent text generation**: **yes, partially**. Output is grammatical-looking, lexically diverse,
  non-repetitive, and respects punctuation and the `Bot:` register. It has learned local English
  syntax.
- **Genuine task performance**: **no.** It cannot answer questions, cannot hold a topic across
  turns, and produces semantically empty continuations. **This model cannot converse**, and no
  result in this report should be read as claiming it can.

---

## 7. Conclusions

1. **SI-HLC's generational selection produced a real, large, consistent improvement in held-out
   language-modelling loss.** −0.60 mean test loss, −41 % mean perplexity, 5/5 seeds better. That
   is a substantially stronger and cleaner result than the MNIST experiment produced, where HLC
   only moved calibration.

2. **The fixed and adaptive curricula (B, C) did not help and slightly hurt.** Both are +0.054
   worse than AdamW, consistently across all seeds, and B and C are nearly identical to each other
   (the adaptive transition rules rarely fire differently from the fixed schedule at this budget).
   So the gain comes specifically from **generational policy selection**, not from the curriculum
   itself. This is a notable contrast with the MNIST run, where the fixed schedule was the best
   condition.

3. **Better loss ≠ language ability.** The 41 % perplexity gain is real, but the model still fails
   every task metric. Low perplexity on a small, narrow, high-formulaicity dialogue corpus means
   the model has learned the corpus's *statistical texture*, not its *meaning*. This is exactly the
   failure mode the constraint "do not claim the model can converse until the evaluation supports
   it" is designed to catch, and it does not support it.

4. **The budget is the binding constraint, not the algorithm.** 2.46 M tokens for a 10 M-parameter
   model is roughly 0.25 tokens per parameter — two orders of magnitude below the Chinchilla-style
   ratio. A 10 M-parameter LM needs tens of millions of tokens before semantics, as opposed to
   local syntax, starts to appear. On this CPU box the achievable budget was the limit, not the
   method.

**Recommended next experiment** (in priority order):

1. **Raise the token budget by an order of magnitude.** DailyDialog's 5.4 M characters is small;
   pair it with a larger permissively-licensed corpus (e.g. CC-BY SQuAD or a public-domain book
   corpus) so a 10 M model can be trained on ~50 M tokens. This is the single highest-value change.
2. **Fix turn-taking explicitly.** Add a `<|bot|>`/`<|endoftext|>` boundary and a hard stop token;
   the current `<|doc|>`-separated packing lets the model run through turn boundaries, which is
   the most visible structural failure in §6.
3. **Keep the generational tournament, drop or re-tune the fixed curriculum.** The evidence here
   says selection helps and the schedule does not; a cleaner ablation would be
   AdamW vs AdamW+replay vs AdamW+generational-selection with no stage schedule at all.
4. **Report task metrics as primary.** Perplexity and QA/F1 together, never perplexity alone.

---

## 8. Artifacts

```
hlc_lm/
├── lm_core.py               # exact-10M LM, config solver, tokenizer, packing, metrics, sampling
├── lm_hlc_train.py          # six stages, adaptive rules, Policy, generational tournament
├── prepare_lm_data.py       # tokenizer training + DailyDialog packing + manifest
├── run_lm_experiment.py     # driver (smoke / screen / main), compute-matched
├── evaluate_lm.py           # independent held-out eval: loss/ppl, QA, multi-turn, memorisation
├── chat_lm.py               # standalone inference script (demo / --prompt / --interactive)
├── analyse_lm.py            # aggregation + paired per-seed differences
├── tokenizer.json           # byte-level BPE, vocab 7618
├── specials.json            # special-token id map
├── dataset_manifest.json    # source, licence, splits, MD5s
├── eval_sihlc_seed1.json    # full evaluation output for the best checkpoint
├── chat_demo.log            # generated demo battery (unedited)
└── runs/main_20261012_113306/
    ├── config.json
    ├── analysis.json        # aggregated means, stds, paired differences
    ├── summary.json
    └── metrics_{condition}_seed{0..4}.json   (20 files, full logs)
```

All 20 checkpoints (model + optimizer + config + policy + seeds) are on the `wise-sage` boxd
machine at `/home/boxd/hlc/lmruns/main_20261012_113306/ckpt_<condition>_seed<N>.pt`
(~40 MB each). The best is `ckpt_sihlc_seed1.pt` (test loss 5.9145, ppl 370.35). Local disk had
only 2.8 GB free, so the checkpoints were left on the machine rather than duplicated.

**Reproduce:**

```bash
# data (DailyDialog parquet mirror; original host is dead)
curl -L -o lmdata/train.parquet \
  https://huggingface.co/datasets/OpenRL/daily_dialog/resolve/main/data/train-00000-of-00001-f151c79abb2c1fd5.parquet
# ... plus validation-...2407eb323af19881.parquet and test-...66dc7d981b70c918.parquet

# tokenizer + packing (writes lmdata/prepared/, verifies MD5s)
python prepare_lm_data.py

# main experiment: 4 conditions x 5 seeds, ~2.6 h on 8 vCPU
OMP_NUM_THREADS=8 python run_lm_experiment.py --mode main --threads 8

# aggregate
python analyse_lm.py lmruns/main_<timestamp>

# independent evaluation of the best checkpoint
python evaluate_lm.py --ckpt lmruns/main_<timestamp>/ckpt_sihlc_seed1.pt \
  --out eval.json

# talk to it
python chat_lm.py --ckpt lmruns/main_<timestamp>/ckpt_sihlc_seed1.pt --interactive
```

Environment: `uv venv --python 3.12 .venv && uv pip install --index-url \
https://download.pytorch.org/whl/cpu torch numpy pyarrow tokenizers`
