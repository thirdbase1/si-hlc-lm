#!/usr/bin/env python3
"""
SI-HLC: Human Life-Cycle Learning algorithm for a 10M-parameter MLP on MNIST.

Six stages with explicit, measurable transition rules:
  1. BIRTH        - learn foundational patterns from easy examples
  2. GROWTH       - gradually increase difficulty and diversity
  3. EXPLORATION  - prioritise uncertain / unfamiliar patterns
  4. REFLECTION   - revisit repeatedly-missed examples
  5. CONSOLIDATION- mix old and new to reduce forgetting
  6. RENEWAL       - evaluate progress/retention/stagnation, choose next stage

Generational learning: weights persist; at RENEWAL, candidate policies are
evaluated from the SAME model + optimizer checkpoint with equal budgets; an
improved candidate replaces the incumbent only if validation supports it.
"""
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
INPUT_DIM = 784
NUM_CLASSES = 10
HIDDEN = (2074, 2034, 2030)   # exactly 10,000,000 trainable params
TARGET_PARAMS = 10_000_000

STAGES = ["BIRTH", "GROWTH", "EXPLORATION", "REFLECTION", "CONSOLIDATION", "RENEWAL"]

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def build_model(hidden: Tuple[int, ...] = HIDDEN) -> nn.Module:
    layers: List[nn.Module] = [nn.Flatten()]
    prev = INPUT_DIM
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, NUM_CLASSES)]
    return nn.Sequential(*layers)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(model: nn.Module, seed: int) -> nn.Module:
    """Kaiming-uniform init (PyTorch default) re-seeded for determinism."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                # Kaiming uniform, fan_in, a=sqrt(5) -- matches PyTorch default
                fan_in = m.weight.shape[1]
                bound = 1.0 / math.sqrt(fan_in)
                m.weight.uniform_(-bound, bound, generator=g)
                if m.bias is not None:
                    m.bias.uniform_(-bound, bound, generator=g)
    return model


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_mnist(data_dir: str):
    """Load MNIST from raw IDX files. Returns uint8 arrays."""
    import gzip

    def read_images(path):
        with gzip.open(path, "rb") as f:
            data = f.read()
        magic = int.from_bytes(data[0:4], "big")
        assert magic == 2051, f"bad image magic {magic} in {path}"
        n = int.from_bytes(data[4:8], "big")
        rows = int.from_bytes(data[8:12], "big")
        cols = int.from_bytes(data[12:16], "big")
        arr = np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows, cols)
        return arr

    def read_labels(path):
        with gzip.open(path, "rb") as f:
            data = f.read()
        magic = int.from_bytes(data[0:4], "big")
        assert magic == 2049, f"bad label magic {magic} in {path}"
        n = int.from_bytes(data[4:8], "big")
        return np.frombuffer(data[8:], dtype=np.uint8)

    xtr = read_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    ytr = read_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    xte = read_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    yte = read_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    return xtr, ytr, xte, yte


def to_tensor(x: np.ndarray, y: np.ndarray, device) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)
    # normalise like MNIST convention
    t = (t - 0.1307) / 0.3081
    return t.to(device), torch.from_numpy(y.astype(np.int64)).to(device)


def split_train_val(
    xtr: np.ndarray, ytr: np.ndarray, val_size: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic train/val split of the official training set."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(xtr))
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]
    return xtr[tr_idx], ytr[tr_idx], xtr[val_idx], ytr[val_idx]


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 1024,
) -> Dict[str, float]:
    """Accuracy, macro-F1, mean NLL loss."""
    model.eval()
    n = len(x)
    correct = 0
    total_loss = 0.0
    nb = 0
    # confusion counts per class: tp, fp, fn
    tp = torch.zeros(NUM_CLASSES, dtype=torch.long)
    fp = torch.zeros(NUM_CLASSES, dtype=torch.long)
    fn = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for i in range(0, n, batch_size):
        xb = x[i : i + batch_size]
        yb = y[i : i + batch_size]
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        nb += len(yb)
        for c in range(NUM_CLASSES):
            p_c = pred == c
            t_c = yb == c
            tp[c] += (p_c & t_c).sum()
            fp[c] += (p_c & ~t_c).sum()
            fn[c] += (~p_c & t_c).sum()
    model.train()
    acc = correct / nb
    prec = tp.float() / (tp + fp).clamp(min=1).float()
    rec = tp.float() / (tp + fn).clamp(min=1).float()
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    macro_f1 = f1.mean().item()
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "nll": total_loss / nb,
        "n": nb,
    }


@torch.no_grad()
def predict_with_uncertainty(
    model: nn.Module, x: torch.Tensor, batch_size: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predicted labels, max-softmax confidence) for all rows."""
    model.eval()
    preds, confs = [], []
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size]
        logits = model(xb)
        p = F.softmax(logits, dim=1)
        conf, pred = p.max(dim=1)
        preds.append(pred.cpu().numpy())
        confs.append(conf.cpu().numpy())
    model.train()
    return np.concatenate(preds), np.concatenate(confs)


# ----------------------------------------------------------------------------
# Experience record
# ----------------------------------------------------------------------------

@dataclass
class Experience:
    """Compact, rolling record of learning signals used for stage decisions."""

    loss_window: List[float] = field(default_factory=list)
    err_window: List[float] = field(default_factory=list)
    val_history: List[Dict[str, float]] = field(default_factory=list)
    # per-example miss counts on the training set (reflection signal)
    miss_counts: Optional[np.ndarray] = None
    # retention: accuracy on a held-out sample of already-seen easy examples
    retention_score: float = 1.0
    uncertainty_mean: float = 0.0
    grad_evals: int = 0
    examples_seen: int = 0
    step: int = 0
    # gradient steps spent on exploratory candidate probes (not lineage training)
    probe_grad_evals: int = 0

    def recent_loss(self, k: int = 50) -> float:
        if not self.loss_window:
            return float("nan")
        return float(np.mean(self.loss_window[-k:]))

    def recent_err(self, k: int = 50) -> float:
        if not self.err_window:
            return float("nan")
        return float(np.mean(self.err_window[-k:]))

    def learning_progress(self, k: int = 50) -> float:
        """Positive when recent loss is below the earlier window (learning)."""
        if len(self.loss_window) < 2 * k:
            return 0.0
        early = float(np.mean(self.loss_window[-2 * k : -k]))
        late = float(np.mean(self.loss_window[-k:]))
        return early - late

    def stagnation(self, k: int = 100) -> float:
        """Small value => loss barely moving (stagnant)."""
        if len(self.loss_window) < 2 * k:
            return 1.0
        a = float(np.mean(self.loss_window[-2 * k : -k]))
        b = float(np.mean(self.loss_window[-k:]))
        return abs(a - b) / (abs(a) + 1e-8)

    def record_batch(self, loss: float, err: float, bs: int):
        self.loss_window.append(float(loss))
        self.err_window.append(float(err))
        self.grad_evals += 1
        self.examples_seen += bs
        if len(self.loss_window) > 2000:
            self.loss_window = self.loss_window[-2000:]
            self.err_window = self.err_window[-2000:]


# ----------------------------------------------------------------------------
# Difficulty scoring (for GROWTH / BIRTH)
# ----------------------------------------------------------------------------

@torch.no_grad()
def score_difficulty(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """Per-example difficulty = cross-entropy loss under the current model."""
    model.eval()
    losses = []
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size]
        yb = y[i : i + batch_size]
        logits = model(xb)
        l = F.cross_entropy(logits, yb, reduction="none")
        losses.append(l.cpu().numpy())
    model.train()
    return np.concatenate(losses)
