# RUNLOG

All runs: 2,000 optimizer steps, CPU, scored with `evaluate.py` on `../data/dev_eval.txt` (bits per byte, lower is better). One variable changed per run; I watched the dev bpb, not the training-loss vibe.

## Results at a glance

| Run | One change | dev bpb | Δ | Kept? |
|----:|------------|:------:|:----:|:-----:|
| 1 | Baseline (constant Adam 3e-4, byte tokenizer) | 2.3718 | — | ref |
| 2 | AdamW + warmup + cosine decay + grad-clip, LR 1e-3 | 2.2447 | -0.127 | yes |
| 3 | Weight tying (head = token embedding) | 2.2731 | +0.028 | yes (frees params, ~neutral) |
| 4 | GPT-2 scaled init (std 0.02, resid /√2L) | 2.4704 | +0.226 | no |
| 4b | Mixed init (emb 0.05, linear 0.02) | 2.4292 | +0.185 | no |
| 5 | Batch 8 → 32 | 1.9629 | -0.282 | yes |
| 6 | Corpus-trained BPE, vocab 1024 | 1.8322 | -0.131 | yes |
| 7 | Spend full param budget (1.9M: vocab 2048, block 256, 5 layers) | 1.835 | +0.003 | no |
| 8 | Batch 32 → 64 | **1.76** | -0.072 | **FINAL** |
| 9 | Scorer-aligned suffix loss (batch 64) | 1.8429 | +0.083 | no |

Headline: **2.3718 → 1.76 bpb, a 26% reduction**, entirely from training efficiency (optimizer, tokenizer, batch) — not from a bigger model. Three ambitious ideas (Runs 4, 7, 9) were tried, lost, diagnosed, and reverted; that reasoning is below.

## Run 1 — Baseline (unmodified starter)

- **Hypothesis:** none; establish the reference number.
- **Config:** byte tokenizer (vocab 256), 4 layers / 4 heads / d=160, block 128, batch 8, Adam lr 3e-4 constant, no warmup, no decay, no clipping, init N(0, 0.05) everywhere, no weight tying. 1,339,840 params.
- **Result:** final train loss 1.7315, **dev bpb 2.3718**. 70 s total (35 ms/step).
- **Conclusion:** loss was still falling steeply at step 2000 — the constant lr 3e-4 is too low / never decays, so the model is undertrained at the step cap. Also every Devanagari char costs 3 byte-tokens, so block 128 covers only ~128 bytes of text. Clear targets: (1) optimizer + schedule, (2) tokenizer, (3) param re-budgeting.

## Run 2 — AdamW + warmup/cosine + clipping, lr 1e-3

- **Hypothesis:** the baseline is undertrained at 2,000 steps; a ~3x higher peak lr with 100-step warmup, cosine decay to 0.1x, weight decay 0.1 (on 2D+ weights only), betas (0.9, 0.95), and grad-norm clip 1.0 should extract much more from the same step budget.
- **Change vs Run 1:** optimizer block only (Adam -> AdamW + schedule + clip). Model/tokenizer untouched.
- **Result:** final train loss 1.5995 (vs 1.7315), **dev bpb 2.2447** (vs 2.3718). Same 35 ms/step.
- **Conclusion:** big win, -0.127 bpb for free. Loss still trending down at the end — the model, not the optimizer, is now the bottleneck. Next: stop wasting params (weight tying) and fix the crude init.

## Run 3 — weight tying only

- **Hypothesis:** tying head to token embedding frees 41K params and usually helps small LMs.
- **Change vs Run 2:** `tie_weights = True`. Nothing else.
- **Result:** final train loss 1.6230, **dev bpb 2.2731** (vs 2.2447). Params 1,298,880.
- **Conclusion:** slightly WORSE on its own. Plausible cause: the crude N(0, 0.05) init is now shared between embedding and output head, and the freed params were not reinvested. Keep tying but fix init next; revisit if it still hurts.

## Run 4 — GPT-2 style init (std 0.02, residual projections / sqrt(2L))

- **Hypothesis:** the textbook scaled init should beat plain N(0, 0.05).
- **Change vs Run 3:** init only.
- **Result:** **dev bpb 2.4704** — much worse.
- **Conclusion:** with only 2,000 steps, small init means small logits and small gradients early; the model spends its whole budget escaping the flat start. Textbook advice assumes 100x more steps.

## Run 4b — embedding std 0.05, linear std 0.02, scaled residual proj

- **Hypothesis:** maybe only the (tied) embedding needed the bigger scale.
- **Result:** **dev bpb 2.4292** — still worse than Run 2/3.
- **Conclusion:** the down-scaled residual projections are the real drag at this step budget. Reverted init entirely to N(0, 0.05). Lesson: at 2,000 steps, "train fast from step 1" beats "clean signal propagation at depth 4".

## Run 5 — batch 8 -> 32

- **Hypothesis:** the caps limit optimizer steps, not tokens; 4x batch = 4x data per step for the same 2,000 steps. Should reduce gradient noise and effectively train on 4x more text.
- **Change vs Run 3 (init reverted):** `--batch 32`.
- **Result:** final train loss 1.3031, **dev bpb 1.9629** (vs 2.2447 best so far). Cost: 129 ms/step, 257 s total — still fine.
- **Conclusion:** biggest single win yet (-0.28 bpb). Wall-clock is the only price, and we have headroom; will consider batch 64 in the final config. Next bottleneck: byte tokenizer wastes the 128-token context (Devanagari = 3 tokens/char).

## Run 6 — byte-level BPE tokenizer, vocab 1024 (batch 32)

- **Hypothesis:** byte tokens waste context. A BPE trained on the corpus (base 256 bytes + 768 merges, so still lossless byte fallback) compresses ~2.9 bytes/token on dev, so block 128 now covers ~370 bytes instead of 128. More real context per prediction should lower bpb even though per-token loss looks higher.
- **Change vs Run 5:** new `tokenizer.py` (BPE) + `train_bpe.py`; vocab 256 -> 1024. Train loss is now per-BPE-token so not comparable to earlier runs; only bpb is.
- **Result:** **dev bpb 1.8322** (vs 1.9629). Params 1,421,760 (bigger embedding). 147 ms/step.
- **Conclusion:** confirmed — bpb is what matters, not token loss. Lossless round-trip verified on dev + Hindi/mixed/empty edge cases. Still 0.58M params under cap. Next: push vocab higher (more context) and spend the param budget.

## Run 7 — scale to ~2M params: vocab 2048, block 256, 5 layers (batch 32)

- **Hypothesis (ambitious, deliberately multi-change):** spend the whole param budget and give the model more context. Vocab 1024 -> 2048 (3.41 bytes/token), block 128 -> 256, layers 4 -> 5. 1,915,360 params, right under the 2M cap.
- **Result:** **dev bpb 1.835** (vs 1.8322 at R6) — no improvement, marginally worse. Cost: 413 ms/step, 826 s total (2.8x R6's wall-clock).
- **Conclusion:** this is the key negative result. Maxing the parameter budget does NOT help under a 2,000-step cap: the larger model + longer context are undertrained, so bpb stalls while wall-clock triples. Confirms the compute/step-optimal intuition — the bottleneck is optimizer steps, not capacity. Reverting to the efficient R6 config (vocab 1024, block 128, 4 layers) and spending effort on things that improve per-step learning (batch size, LR) instead of raw size.

## Run 8 — batch 32 -> 64  ***(FINAL CONFIGURATION, this is `ckpt.pt`)***

- **Hypothesis:** since steps are capped, the cheapest way to feed more data into the fixed step budget is a bigger batch. R5 already showed 8 -> 32 helped a lot; test 64.
- **Change vs R6:** `--batch 64` (vocab 1024, block 128, 4 layers, everything else unchanged).
- **Result:** **dev bpb 1.76** (vs 1.8322). 298 ms/step, 597 s. 1,421,760 params, 2,000 steps.
- **Conclusion:** another clear win (-0.07) and the best measured configuration, so this is the submitted `ckpt.pt`. Confirms the lever is tokens-per-step, not model size: over 2,000 steps, batch 8/32/64 see ~2.05M / ~8.19M / ~16.38M target tokens respectively. Larger batch would likely help further (with a re-tuned LR), but was capped here by CPU wall-clock — see "What I'd try next".

## Run 9 — scorer-aligned suffix loss (batch 64)

- **Hypothesis:** `evaluate.py` only scores targets with >= block/2 left context (positions 63-126 of each 128-token window) and never scores position 127. Training loss on *all* positions therefore spends half its gradient on short-context predictions the scorer ignores. Masking training loss to positions 63+ should make every supervised target match the eval-time context distribution and lower bpb.
- **Change vs R8:** added `--suffix_loss` (CE on `logits[:, 63:]` only); batch 64, LR 1e-3, everything else identical.
- **Result:** **dev bpb 1.8429** (vs 1.76 at R8) — worse. 293 ms/step, 587 s.
- **Conclusion:** the alignment idea backfired because it halves supervised targets per step (64 positions vs 128), and under the 2,000-step cap raw token throughput beats context-alignment. To test the idea fairly you'd need to double batch to keep target count constant, but that doubles wall-clock. Reverted; **final submission stays at R8 (1.76 bpb).** Third negative result confirming the same theme: at 2,000 steps, maximize useful gradient signal per step, not cleverness.

## Analysis: what actually moved the metric

Ranked by impact: **batch size** (-0.35 across Runs 5+8) > **optimizer/schedule** (-0.13) > **tokenizer** (-0.13). Every winner increases *useful gradient signal per optimizer step*, which is the one resource the caps actually constrain. Every loser (bigger model, textbook init, scorer-aligned masking) either spends the fixed step budget on undertrained capacity or reduces targets-per-step. The unifying principle: with steps capped at 2,000, the objective is step-efficiency, and "bigger / longer / cleverer" only helps if it doesn't cost gradient throughput.

## What I'd try next (given more step or wall-clock budget)

1. **Batch 96–128 with a re-tuned peak LR** (~1.25e-3, shorter warmup) — Run 8 showed batch is the dominant lever and it was still improving; I stopped at 64 only for wall-clock. This is the single most likely further win.
2. **BPE vocab sweep (768 / 1280 / 1536)** — 1024 was a first guess; there is a bias/variance sweet spot between sequence compression and embedding-parameter cost that I did not fully search.
3. **RMSNorm + no-bias linears + SwiGLU MLP**, one at a time — modern components that often improve small models at equal params; each is a clean single-variable test.
4. **Coverage-balanced sampling** — replace random-with-replacement window starts with shuffled full-corpus coverage, so all 2,000 steps see maximally diverse data.

Each is a single-variable experiment with a concrete hypothesis, consistent with the discipline above.
