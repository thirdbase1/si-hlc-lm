# SI-HLC Language Model

Does a human-life-cycle-inspired (HLC) training curriculum improve a small causal language
model's language learning? A 10 M-parameter decoder-only Transformer trained **from scratch**
on DailyDialog, with four compute-matched conditions.

**Result: yes for held-out loss (5/5 seeds), no for conversation.** See `REPORT.md`.

| Condition | Test loss | Test ppl | Paired Δ vs AdamW | Win rate |
|---|---|---|---|---|
| A `adamw` | 6.9729 ± 0.0767 | 1069.9 | — | — |
| B `hlc_fixed` | 7.0271 ± 0.0773 | 1129.5 | +0.0542 | 0/5 |
| C `hlc_adaptive` | 7.0270 ± 0.0773 | 1129.3 | +0.0541 | 0/5 |
| D `sihlc` | **6.3716 ± 0.4263** | **627.6** | **−0.6013** | **5/5** |

But the best model still scores **QA token-F1 0.089 / exact match 0.000** and produces
well-formed English with no semantic content. It **cannot converse**.

## Model

Decoder-only causal Transformer, exactly **10,000,000 trainable parameters** (asserted at
runtime and cross-checked against a closed form).

```
vocab=7618  d_model=320  n_layers=6  ffn=1280 (4x)  n_heads=5  pos_max=512  tied embeddings
```

Vocab 7618 is *derived* from the parameter constraint: with tied embeddings the count is
`V*d + P*d + L*(4d^2 + 4d + 2*d*ffn)`, so `V = (10_000_000 - 6*1_232_960 - 164_480) / 320 = 7618`.
The tokenizer was then trained to that size. Initialisation is PyTorch default, re-seeded per
run; **no pretrained weights anywhere**.

## Data

DailyDialog (Li et al., IJCNLP 2017), from the `OpenRL/daily_dialog` HuggingFace parquet mirror
(the original author host is dead and returns HTML). **Licence: CC-BY-NC-SA-4.0 —
non-commercial research use only.** 11,118 / 1,000 / 1,000 dialogues → 6,109 / 563 / 551 packed
256-token blocks. Byte-level BPE, **trained on the train split only** so val/test text cannot
leak into the vocabulary.

## Quickstart

```bash
# data
curl -L -o lmdata/train.parquet \
  https://huggingface.co/datasets/OpenRL/daily_dialog/resolve/main/data/train-00000-of-00001-f151c79abb2c1fd5.parquet
# + validation-00000-of-00001-2407eb323af19881.parquet
# + test-00000-of-00001-66dc7d981b70c918.parquet

python prepare_lm_data.py                          # tokenizer + packing + MD5s
OMP_NUM_THREADS=8 python run_lm_experiment.py --mode main --threads 8   # ~2.6 h on 8 vCPU
python analyse_lm.py lmruns/main_<timestamp>
python evaluate_lm.py --ckpt lmruns/main_<timestamp>/ckpt_sihlc_seed1.pt --out eval.json
python chat_lm.py --ckpt lmruns/main_<timestamp>/ckpt_sihlc_seed1.pt --interactive
```

Environment: `uv venv --python 3.12 .venv && uv pip install --index-url \
https://download.pytorch.org/whl/cpu torch numpy pyarrow tokenizers`

## The six stages

BIRTH → GROWTH → EXPLORATION → REFLECTION → CONSOLIDATION → RENEWAL (× generations).
Sampling is biased by live per-block signals: difficulty (mean CE), uncertainty (mean entropy),
miss (EMA token error), replay pool. Transitions use **measurable thresholds** on the LM loss
scale (untrained ≈ ln V = 8.94; trained ≈ 4.5–6.9), not the MNIST 0–1 scale.

RENEWAL runs a generational policy tournament: snapshot weights **and** optimizer state, probe
3 mutated candidates on an equal budget, score on **validation** loss, keep a candidate only if
it strictly wins, then restore the pre-tournament checkpoint regardless. Weights persist across
generations — there is no restart from scratch.

## Compute matching

All conditions: exactly **600 steps × 4,096 tokens = 2,457,600 lineage tokens** (verified in
every metrics file). Condition D's probe steps are excluded from the lineage budget (state is
restored and counters rolled back) and reported separately as `probe_steps`. This is why D costs
~77 % more wall-clock.

## Files

| File | Purpose |
|---|---|
| `lm_core.py` | exact-10M LM, config solver, tokenizer, packing, metrics, sampling |
| `lm_hlc_train.py` | six stages, adaptive rules, Policy, generational tournament |
| `prepare_lm_data.py` | tokenizer training + DailyDialog packing |
| `run_lm_experiment.py` | driver (smoke / screen / main) |
| `evaluate_lm.py` | independent eval: loss/ppl, QA, multi-turn, memorisation |
| `chat_lm.py` | standalone inference script |
| `analyse_lm.py` | aggregation + paired per-seed differences |
| `REPORT.md` | full report with honest limitations |

Checkpoints (20 × ~40 MB) live on the boxd machine at
`/home/boxd/hlc/lmruns/main_20261012_113306/ckpt_<condition>_seed<N>.pt`. All 20 verified:
strict `state_dict` load, exactly 10,000,000 params, stored `params` field agrees with an
independent count.
