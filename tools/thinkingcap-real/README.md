# ThinkingCap REAL - pure-torch cap harness for DeepSeek-V4-Flash ablit

Pure-torch forward + train + gate harness for the DeepSeek-V4-Flash
(MXFP4 experts + FP8 dense) abliterated checkpoint. This sidesteps the
stock tooling entirely: an FP8 dequant-on-the-fly LUT loader (bit-equal
to the served weights), a pure-torch full engine (forward, greedy
generate, LoRA attach with a fast bf16 path), and the ThinkingCap cap
pipeline: on-policy regen, oracle filter, LoRA train, merge, and a
before/after A/B gate.

No ms-swift, no Megatron, no cloud framework needed. Runs on a single
consumer GPU or unified memory (peak about 5.19GB per layer on a 128GB
unified-memory machine) at zero-cost-class compute.

## Contents

- `full_loader.py` - FP8 dequant-on-the-fly loader: FP8 dense weights
  bit-equal to the served checkpoint, MXFP4 experts via per-32-group E8M0
  scales intact, streaming tensor reader (3.58GB/layer for 154GB stacks).
- `dsv4_full.py` - pure-torch engine: route-then-load, forward, greedy
  generate, `attach_lora` (with the key-prefix strip fix cited in-file),
  fast-bf16 LoRA path. Line-cited notes mirror the reference engine
  fetched from the reference tree (see `dsv4_full.py` header).
- `tc_lora_train.py` - the ThinkingCap cap train: R=8, alpha=16, 258
  adapter sites on language-layer projections + the shared expert,
  everything else frozen (drafter-anchor lesson: the real run freezes
  vision/MTP the same way). LoRA semantics exact for the merge rehearsal:
  W_eff = W + (alpha/r) * (B @ A), A kaiming, B zero -> step 0 is
  bit-identical to the base.
- `tc_batch_gen.py` / `tc_holdout_gate.py` - on-policy regen + the
  before/after gate: thinking-ON greedy batched A/B (thinking regime),
  mean completion tokens (all + correct-only), first-answer match,
  hit-cap count, the merged report.
- `smoke_forward.py` - engine smoke gates: bit-equality of the loader vs
  the served weights, top-5 token plausibility (multi-char pieces, not
  byte junk), greedy coherence, determinism across two runs.
- `fp4_authority.py` - the unused cross-check module (kept for the
  FP4-experts LUT authority question).
- `prep/` - recipes: the alpha sweep plan, the ablation fallback doc, and
  `gen_problems_podscale.py` (the 500-2000 problem generator for the
  full-scale regen).
- `pod-bundle/` - the bench bundle: `pod_bench.py` (thinking-ON greedy
  batched A/B, args --arm base|cap --n 48 --budget 512 --open-budget 768),
  `pod_run.sh` (smoke + dual-GPU arms + merge summary), `lora.safetensors`
  (the pilot cap artifact, 95MB, 258 adapter sites), `tensor_meta.json`,
  `lora_path_control.py` (the behavioral control: forward_batch base vs
  attached; a BATCHED-PATH LOGIT MAXDIFF decides whether adapters are
  live - a count printed is not evidence).
- `data/` - holdout problems + regen receipts from the pilot scale.
- `PROGRESS.md` - the sanitized progress log.

## Results (pilot scale, 40-60 problems, $0-class compute)

- Engine: FP8 loader bit-equal + MXFP4-experts LUT maxdiff 0.0; smoke
  ALL PASS (coherent greedy, deterministic).
- TC cap train: R=8 alpha=16, 258 sites, step-0 bit-identity, trainable
  artifact 95MB.
- Behavioral control: attach_lora key-prefix fix (keys stored as
  `layers.N.site.weight` but the lookup wanted `site.weight` -> adapters
  never applied in ANY path). Fix = strip the prefix after extracting
  `li`. Control receipts: logit maxdiff 0.0 (broken) -> 20.99 (fp32
  live) -> 20.86 (fast bf16 path). THE LESSON: a count printed is not
  evidence - a behavioral control (logit diff vs baseline) is.

## The author's recipe reference (the scale this harness targets)

khudgins/ornith-thinking-cap (MIT license): the hotdogs method, 369 SFT +
244 DPO samples, proven on 9B/35B dense models on one DGX-class machine.
The MoE lesson: attention-only LoRA caps at about -24% percent - include
routed experts. The regen class: few hundred to 2k verifiable problems,
3 samples each, oracle-verified, shortest-correct SFT + short-vs-verbose
DPO. `prep/gen_problems_podscale.py` generates that scale.

## Usage (cloud GPU or local)

```
docker run --rm --gpus all -v .:/wd -v <checkpoint-dir>:/model:ro -w /wd \
  <python-image> python3 smoke_forward.py
python3 tc_batch_gen.py --n 500 --n-samples 3   # full-scale regen
python3 tc_lora_train.py --data data/regen.jsonl
python3 tc_holdout_gate.py                       # before/after gate
```

Machine names, addresses and paths in these files are generic sanitized
forms; plug in your own. No credentials, tokens, or vendor specifics are
included on purpose - see the privacy note in the repository root.
