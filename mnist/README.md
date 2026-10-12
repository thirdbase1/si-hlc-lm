# SI-HLC — Human Life-Cycle Learning experiment

Fresh 10,000,000-parameter MLP trained from scratch on MNIST under four matched conditions,
testing whether a human-life-cycle-inspired curriculum (BIRTH → GROWTH → EXPLORATION →
REFLECTION → CONSOLIDATION → RENEWAL) improves learning efficiency and generalisation.

**Read [`REPORT.md`](REPORT.md) for the full results and honest conclusions.**

## Quick summary

| Condition | Test acc | Test NLL (mean ± std) | Grad evals |
|---|---|---|---|
| A `adamw` (baseline) | 0.9752 ± 0.0015 | 0.1140 ± 0.0148 | 3000 |
| B `hlc_fixed` | 0.9763 ± 0.0021 | **0.0981 ± 0.0082** | 3000 |
| C `hlc_adaptive` | 0.9757 ± 0.0027 | 0.1117 ± 0.0127 | 3000 |
| D `sihlc` (+generational) | 0.9755 ± 0.0011 | 0.1080 ± 0.0040 | 3000 |

Bottom line: the curriculum reliably improves **loss/calibration** (−14 % NLL, half the variance)
but not **accuracy**; adaptivity and generational selection add cost without adding accuracy.

## Files

- `hlc_core.py` — model (exactly 10,000,000 params), MNIST loader, metrics (acc / macro-F1 / NLL),
  difficulty & uncertainty scoring, `Experience` record
- `hlc_train.py` — the six stages, adaptive transition rules, `Policy`, mutation, and the
  generational tournament at RENEWAL
- `run_experiment.py` — driver with `smoke` / `screen` / `main` modes
- `analyse.py` — aggregation, paired per-seed differences vs. baseline
- `verify.py` — independently reloads a checkpoint and re-evaluates it
- `show_stages.py` — prints the stage-transition log
- `runs/main_20261012_080850/` — config, report, summary, 20 metric files, best SI-HLC checkpoint

## Where the runs live

Training ran on the boxd machine `wise-sage` (8 vCPU / 32 GiB / CPU-only). All 20 checkpoints
(model + optimizer + config + seeds) are at:

```
/home/boxd/hlc/runs/main_20261012_080850/ckpt_<condition>_seed<0-4>.pt
```

The best SI-HLC checkpoint (seed 2, test acc 0.9769) is also copied locally to
`runs/main_20261012_080850/ckpt_sihlc_seed2.pt`.

## Reproduce

```bash
# 1. get the code onto the machine
for f in hlc_core.py hlc_train.py run_experiment.py; do
  boxd machine cp $f wise-sage:/home/boxd/hlc/$f
done

# 2. MNIST (canonical IDX; MD5s verified against the reference checksums)
boxd machine exec wise-sage -- sh -c 'cd /home/boxd/hlc && mkdir -p data && \
  BASE=https://ossci-datasets.s3.amazonaws.com/mnist && \
  for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
           t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do \
    curl -sS -L -o data/$f $BASE/$f; done && md5sum data/*.gz'

# 3. smoke test (fast, verifies all plumbing)
boxd machine exec wise-sage -- sh -c 'cd /home/boxd/hlc && . .venv/bin/activate && \
  python run_experiment.py --mode smoke'

# 4. main experiment (4 conditions x 5 seeds, ~30 min)
boxd machine exec wise-sage -- sh -c 'cd /home/boxd/hlc && . .venv/bin/activate && \
  python run_experiment.py --mode main --budget 3000 --generations 3'

# 5. aggregate + verify
boxd machine exec wise-sage -- sh -c 'cd /home/boxd/hlc && .venv/bin/python analyse.py runs/main_*'
boxd machine exec wise-sage -- sh -c 'cd /home/boxd/hlc && .venv/bin/python \
  verify.py runs/main_*/ckpt_sihlc_seed2.pt'
```

Environment: `uv venv --python 3.12 .venv && uv pip install --index-url \
https://download.pytorch.org/whl/cpu torch numpy`
