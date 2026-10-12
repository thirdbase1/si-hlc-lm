#!/usr/bin/env python3
"""
prepare_lm_data.py -- tokenizer training + DailyDialog packing.

Trains a byte-level BPE tokenizer ON THE TRAINING SPLIT ONLY (so no validation
or test text leaks into the vocabulary), then packs all three splits into
fixed-length blocks.

Writes to lmdata/prepared/:
  tokenizer.json, tokenizer_config.json, specials.json
  train_blocks.npy, val_blocks.npy, test_blocks.npy
  doc_bounds_{split}.npy   (dialogue -> token offsets, for reporting)
  manifest.json            (vocab size, counts, checksums, config)
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lm_core import (BOT_TAG, DOC_SEP, PAD, SEQ_LEN, SPECIALS, USER_TAG,
                     analytic_params, format_dialogue, load_dailydialog)

DATA_DIR = os.environ.get("LM_DATA_DIR", "/home/boxd/hlc/lmdata")
OUT = os.path.join(DATA_DIR, "prepared")
VOCAB_TARGET = 7618           # solves to EXACTLY 10,000,000 params at L=6


def train_tokenizer(train_texts):
    """Byte-level BPE on the training split only."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=VOCAB_TARGET,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    # ByteLevel is whitespace-preserving by design; we train on the formatted
    # dialogues so the model sees natural turn boundaries.
    texts = [format_dialogue(d) for d in train_texts]
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = load_dailydialog(DATA_DIR)
    print("raw dialogues:", {k: len(v) for k, v in raw.items()})

    tok = train_tokenizer(raw["train"])
    vocab = tok.get_vocab_size()
    print("trained tokenizer vocab:", vocab)

    # exact-param check must hold for the ACTUAL vocab, not the target
    for nl in range(2, 14):
        if analytic_params(vocab, 320, nl, 1280, 512, True) == 10_000_000:
            print(f"  -> exact 10M config: n_layers={nl}")
            break
    else:
        raise SystemExit(f"vocab {vocab} does not admit an exact 10M config")

    doc_sep_id = tok.token_to_id(DOC_SEP)
    pad_id = tok.token_to_id(PAD)
    print(f"doc_sep={doc_sep_id} pad={pad_id}")

    manifest = {
        "dataset": "DailyDialog (Li et al., IJCNLP 2017)",
        "source_repo": "OpenRL/daily_dialog (HF parquet mirror of the original "
                       "ijcnlp_dailydialog release)",
        "original_url": "http://yanran.li/files/ijcnlp_dailydialog.zip (dead)",
        "license": "CC-BY-NC-SA-4.0 (non-commercial research use)",
        "tokenizer": "byte-level BPE (HuggingFace tokenizers), trained on the "
                     "TRAINING SPLIT ONLY",
        "vocab_size": vocab,
        "seq_len": SEQ_LEN,
        "doc_sep_id": doc_sep_id,
        "pad_id": pad_id,
        "formatting": f"alternating '{USER_TAG}' / '{BOT_TAG}' turns, one "
                      f"dialogue per line-block, '{DOC_SEP}' between dialogues",
        "splits": {},
    }

    for split in ["train", "validation", "test"]:
        texts = [format_dialogue(d) for d in raw[split]]
        enc = tok.encode_batch(texts)
        ids = [e.ids for e in enc]
        # stream + chunk
        stream = []
        bounds = []
        for d in ids:
            start = len(stream)
            stream.extend(d)
            stream.append(doc_sep_id)
            bounds.append((start, len(stream)))
        n_blocks = len(stream) // SEQ_LEN
        usable = n_blocks * SEQ_LEN
        arr = np.array(stream[:usable], dtype=np.int64)
        blocks = arr.reshape(n_blocks, SEQ_LEN)
        np.save(os.path.join(OUT, f"{split}_blocks.npy"), blocks)
        np.save(os.path.join(OUT, f"doc_bounds_{split}.npy"),
                np.array(bounds, dtype=np.int64))

        # checksums for reproducibility
        digest = hashlib.md5(arr.tobytes()).hexdigest()
        manifest["splits"][split] = {
            "dialogues": len(texts),
            "blocks": int(n_blocks),
            "tokens_packed": int(usable),
            "tokens_dropped": int(len(stream) - usable),
            "md5_blocks": digest,
        }
        print(f"{split:11s} dialogues={len(texts):6d} blocks={n_blocks:6d} "
              f"tokens={usable:,} md5={digest[:12]}")

    tok.save(os.path.join(OUT, "tokenizer.json"))
    with open(os.path.join(OUT, "specials.json"), "w") as f:
        json.dump({"doc_sep": DOC_SEP, "pad": PAD,
                   "user_tag": USER_TAG, "bot_tag": BOT_TAG,
                   "doc_sep_id": doc_sep_id, "pad_id": pad_id}, f, indent=2)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nwrote manifest + tokenizer to", OUT)


if __name__ == "__main__":
    main()
