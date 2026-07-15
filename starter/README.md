# 2,000-Step LLM Speedrun — final submission

Final result: **1.76 dev bits-per-byte** (baseline 2.3718, ~26% lower), within all caps
(2,000 steps, <2M params, CPU only, provided corpus only, pure PyTorch).

## Reproduce the final configuration

All commands run inside this folder (`starter/`).

1. (Only if `bpe_merges.json` is missing) train the byte-level BPE tokenizer on the corpus:

```
python train_bpe.py --data ../data/train_corpus.txt --vocab_size 1024
```

2. Train the final model (2,000 steps, batch 64, AdamW + warmup/cosine, LR 1e-3):

```
python train.py --data ../data/train_corpus.txt --steps 2000 --lr 1e-3 --batch 64 --out ckpt.pt
```

3. Score (this is the official, unmodified evaluation command):

```
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```

## Expected numbers

- Tokenizer vocabulary: **1,024** (256 raw bytes + 768 corpus-learned merges), ~2.9 bytes/token
- Model parameters: **~1,421,760** (< 2,000,000 cap)
- Optimizer steps recorded in `ckpt.pt`: **2,000**
- Dev bits-per-byte: **~1.76**

## Files

- `ckpt.pt` — final checkpoint (records step count)
- `model.py` — GPT (tied embeddings, LayerNorm, GELU MLP)
- `train.py` — trainer with AdamW, warmup+cosine schedule, grad clipping, CLI overrides
- `tokenizer.py` — lossless byte-level BPE with byte fallback; loads `bpe_merges.json`
- `train_bpe.py` — trains the BPE merges on `train_corpus.txt` only
- `bpe_merges.json` — the learned merges (loaded at train and eval time)
- `evaluate.py` — official scorer, unchanged from the handout
- `RUNLOG.md`, `NOTES.md`, `SUMMARY.html` — write-ups

The caps and the `evaluate.py` interface are fixed; everything else was changed deliberately and logged in `RUNLOG.md`.
