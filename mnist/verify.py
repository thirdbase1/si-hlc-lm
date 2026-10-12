#!/usr/bin/env python3
"""Independent verification of the saved SI-HLC checkpoint."""
import glob
import json
import sys

import torch

from hlc_core import TARGET_PARAMS, build_model, count_params, evaluate, load_mnist, split_train_val, to_tensor

path = sys.argv[1]
ck = torch.load(path, map_location="cpu", weights_only=False)

print(f"checkpoint: {path}")
print(f"  condition : {ck['condition']}")
print(f"  seed      : {ck['seed']}")
print(f"  params    : {ck['params']:,}  (target {TARGET_PARAMS:,}, exact={ck['params']==TARGET_PARAMS})")

m = build_model()
missing, unexpected = m.load_state_dict(ck["model_state"], strict=False)
print(f"  load_state_dict: missing={list(missing)} unexpected={list(unexpected)}")
n = count_params(m)
print(f"  rebuilt model trainable params: {n:,}  exact={n==TARGET_PARAMS}")

# independent re-evaluation on the official MNIST test set
xtr, ytr, xte, yte = load_mnist("data")
_, _, xv, yv = split_train_val(xtr, ytr, 10000, ck["seed"])
xte_t, yte_t = to_tensor(xte, yte, torch.device("cpu"))
xv_t, yv_t = to_tensor(xv, yv, torch.device("cpu"))
m.eval()
te = evaluate(m, xte_t, yte_t, batch_size=1024)
va = evaluate(m, xv_t, yv_t, batch_size=1024)
print(f"  RE-EVALUATED test : acc={te['accuracy']:.4f} macroF1={te['macro_f1']:.4f} nll={te['nll']:.4f}")
print(f"  RE-EVALUATED val  : acc={va['accuracy']:.4f} macroF1={va['macro_f1']:.4f} nll={va['nll']:.4f}")

r = ck["result"]
print(f"  reported test     : acc={r['test_acc']:.4f} macroF1={r['test_f1']:.4f}")
print(f"  reported val      : acc={r['val_acc']:.4f} macroF1={r['val_f1']:.4f}")
ok = abs(r["test_acc"] - te["accuracy"]) < 5e-4
print(f"  MATCH: {ok}")
