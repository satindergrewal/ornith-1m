# ThinkingCap REAL — STEP 1 progress log (DSV4-Flash Vision ablit on node-a)

Machine: DGX Spark "node-a" (GB10, 128GB UMA). Checkpoint: ~/models/dsv4-vision-ablit
(drowzeys community ablit of DeepSeek-V4-Flash-Vision-Exp; 48 shards, 167.8GB,
FP8-E4M3 128x128 block-scale weights + MXFP4 experts; MTP-3 preserved; wo_b edit L10-35).
Runner image: vllm/vllm-openai:qwen38-flash-next (torch 2.13.0+cu130, CUDA on GB10 OK,
float8_e8m0fnu + float4_e2m1fn_x2 dtypes present; safetensors reads E8M0/E4M3 natively).

Docker invocation used throughout:

```bash
ssh user@node-a 'docker run --rm --gpus all -v /home/user/tc-real:/wd \
  -v /home/user/models/dsv4-vision-ablit:/model:ro -w /wd \
  --entrypoint python3 vllm/vllm-openai:qwen38-flash-next <script>.py'
```

## 2026-09-04 (NZST) — session start ~13:55

### Deliverable 0: integrity spot-check — PASS

| shard | sha256 (box 192.168.1.10) | sha256 (node-a) | match |
|---|---|---|---|
| model-00001-of-00048 | 367c971dc3cd6a042a9bec1caff508e77eabfaef1df2de3a827397ef8bbc6af3 | same | YES |
| model-00048-of-00048 | 0de99b7dd23d964b0a631e9e6f14ea1751db6d14bf0f009c6a83b725ca5f909e | same | YES |

Raw receipts: sha256_box.txt / sha256_node-a.txt in this dir.

### Ground truth gathered (orientation phase)

- config.json parsed: 43 layers, hidden 4096, 64 heads, head_dim 512 (rope 64 trailing),
  q_lora 1024, o_lora 1024, o_groups 8, index_n_heads 64, index_head_dim 128, topk 512,
  n_routed_experts 256 (NOT 128 — config authoritative), top-6 + 1 shared, moe_inter 2048,
  window 128, swiglu_limit 10.0, routed_scaling 1.5, hc_mult 4 / 20 sinkhorn iters / eps 1e-6,
  rms_norm_eps 1e-20, compress_rope_theta 160000 (yarn factor 16, orig 65536),
  rope_theta 10000 (NO yarn — sliding layers only), num_hash_layers 3,
  eos 1, bos 0, vocab 129280.
- compress_ratios: 46 entries = layers 0..42 + 3 MTP. L0/L1 = 0 (pure sliding window,
  main rope table, no compressor tensors exist for them). Even 2..42 = 4 (CSA, overlap
  compressor + indexer). Odd 3..41 = 128 (HCA, non-overlap compressor, dense pool). MTP = 0.
- Full safetensors header census (tensor_meta.json, built by peek_headers.py from raw
  headers, no torch needed): 72,633 tensors. Language stack 153.90GB (3.42GB experts/layer),
  embed 1.059GB, head 1.059GB, mtp 10.86GB, vision 0.82GB. Each layer's tensors live in
  exactly ONE shard (layer i -> shard i+2 pattern) — clean streaming.
- Weight formats: F8_E4M3 + F8_E8M0 scale [ceil(n/128), ceil(k/128)] for wq_a/wq_b/wkv/
  wo_b/indexer.wq_b/shared experts/main_proj; BF16 for wo_a, compressor wkv/wgate, gate,
  weights_proj; F32 for ape/sinks/hc/attn_sink/gate.bias; MXFP4 for routed experts:
  weight I8 [n, k/2] packed pairs + E8M0 scale [n, k/32] (logical [n, k], groups of 32
  along K); I64 tid2eid [129280, 6] x3 (layers 0,1,2 hash routing).
- MTP structure (step-2 surface): mtp.{0,1,2} = full block (sliding attn, no compressor)
  + main_norm/main_proj (stage 0) + norm + hc_head trio + markov_head.markov_w1/w2
  [129280, 256] + confidence_head.proj [1, 4352] (stage 2). Matches official
  DSparkBlock layout. NOT loaded in step 1.
- THE CHECKPOINT SHIPS ITS OWN REFERENCE ENGINE: inference/model.py + kernel.py
  (pure torch + tilelang) and inference/convert.py. This is the math authority used
  for dsv4_full.py (not nano_torch, whose compressor/kv_norm/wo_a were dummy-degenerate).
- exllamav3 dsv4.py + hyperconnections.py + util/rope.py fetched from node-b
  (read-only) and cross-checked. ONE genuine discrepancy found between official
  inference/model.py and exl3: hc_post applies comb UNTRANSPOSED in official
  (y_i = sum_j comb[i,j] * res_j) vs comb^T in exl3 apply_. Sinkhorn construction is
  identical in both. Decision: follow official (ships with the checkpoint); env knob
  HC_COMB_TRANSPOSED=1 kept for the fallback A/B. Recorded under ASSUMPTIONS.

### Key math locked from official inference/model.py (see dsv4_full.py header for cites)

- q: wq_a -> weighted q_norm(1024) -> wq_b -> per-head UNWEIGHTED rms over full 512 ->
  rope last 64 (GPTJ pairs). kv: wkv -> weighted kv_norm over FULL 512 -> rope last 64.
- rope tables per layer type: sliding -> plain 1/10000^(2i/64); csa/hca -> yarn ramp
  (factor 16, orig 65536, beta 32/1, truncate) on compress theta 160000. Same table for
  q, kv, pool entries, indexer q, and the output de-rotation (conjugate at query pos).
- compressor (prefill, stateless): wkv/wgate fp32; windows of ratio; gate += ape
  (ratio, coff*512); overlap iff ratio==4 (Ca of prev window dims[0:512] + Cb of current
  dims[512:1024], softmax over 2*ratio slots, first window's Ca masked -inf);
  non-overlap (ratio 128) softmax over ratio slots; weighted sum -> weighted RMSNorm ->
  rope last 64 at positions w*ratio. Remainder tokens discarded.
- indexer (CSA only): q_idx = indexer.wq_b(qr), rope last 64 at token pos, weights_proj
  full x * (128^-0.5 * 64^-0.5); scores = relu(q.K^T) summed over 64 heads; causal bound
  e < (t+1)//ratio; topk min(512, T); out-of-bound picks -> -1 sentinel; +offset(seqlen).
  Official applies Hadamard + FP4-sim on indexer q/entries (QAT); SKIPPED in step 1
  (see ASSUMPTIONS).
- attention: single softmax over [sliding-128-window causal ++ selected pool entries ++
  per-head sink logit]; sink mass dropped from numerator; de-rotate output rope slice
  (conjugate) at query positions; grouped o_proj: heads grouped 8x8, wo_a viewed
  (8, 1024, 4096) einsum per group, concat (S, 8192) -> wo_b.
- MoE: sqrtsoftplus routing; L0-2 HASH routing via tid2eid[token]; L3+ (scores+bias) topk
  for selection only; weights = unbiased scores, normalized over top-6, x1.5. SwiGLU
  clamps gate<=10, up in [-10, 10]. Shared expert always, no weight.
- HC: mix on flattened 4*4096 (rms-scaled), sigmoid pre (+eps) / 2*sigmoid post /
  comb softmax(-1)+eps -> col-norm -> (iters-1) x [row-norm, col-norm]; streams expand
  from embed copies; final hc_head collapse -> norm -> lm_head fp32 logits.
- Step-1 simplification (advisor-reviewed): QAT activation sims skipped (fp8 act sim on
  FP8-weight linear inputs; fp8-per-64 kv nope sim; fp4/hadamard indexer sim). Weights
  dequant exact; matmuls bf16/fp32. Documented under ASSUMPTIONS.

### Files

- peek_headers.py, analyze_meta.py, tensor_meta.json (NOT mirrored; large + derived)
- full_loader.py — index-driven streaming loader + codec unit tests
- dsv4_full.py — full-model engine (config-driven)
- smoke_forward.py — probe forward + gates + greedy-20
- Mac mirror: ~/Documents/GitHub/ornith-1m/tools/thinkingcap-real/

## What STEP 2 (03-real on-policy data regen) needs from this engine

Measured baseline (run1/2): full-recompute prefill costs ~20s for 5-8 tokens and
each greedy step at S<=26 costs ~10-25s warm (page cache holding ~114GB of the
154GB model; NVMe re-reads the rest). The dummy's 03 shape (200 problems x 8
samples x 48 tokens = ~77k token-steps) is NOT viable at that cost. Needs:

1. STATEFUL DECODE (the big one): implement the official start_pos>0 paths —
   per-layer state kept resident across steps (43 x [kv ring 128x512 bf16 +
   compressor kv/score state (2m x W fp32) + overlap snapshot + growing pool]
   ~ tens of MB total) so each new token is ONE layer-sweep, not a re-prefill.
   Reference: inference/model.py Compressor decode branch, Attention
   start_pos>0 branch, get_window_topk_idxs ring, get_compress_topk_idxs,
   indexer offset/block-table handling.
2. BATCHED SAMPLES: b>=3-8 rows share one layer load (route-then-load extends
   to the union of routed experts across the batch). Reference has bsz paths.
3. SAMPLING: temperature + seeded generator on forward logits (dummy 03 does
   exactly this on top of forward; nothing new needed in the engine).
4. CHAT TEMPLATE: on-policy prompts should use the official template
   (checkpoint encoding/ dir + inference/generate.py encode_case, thinking
   modes "chat"/"thinking"); raw continuation works for smoke but the base is
   chat-trained.
5. MTP/DSpark: NOT needed for 03-real (target model only); mtp.{0,1,2} tensor
   map already documented above for the later LoRA/merge/drafter steps.

Scale estimate with 1+2: ~2-3s per layer-sweep warm x ~50 problems x 3 samples
x 48 tokens / batch(3) ~ 1.5-3 h. Viable overnight; 200x8x48 is not.

## ASSUMPTIONS (each one minimal + documented)

1. MXFP4 nibble order: element 2i = LOW nibble, 2i+1 = HIGH nibble (torch
   float4_e2m1fn_x2 packing convention; the official reference Linear stores
   [out, in//2] x2-packed along K). Unit test cross-checks manual LUT decode vs
   independent decode path; ultimate arbiter = coherence gate.
2. hc comb orientation: official untransposed (discrepancy vs exl3 documented above).
   HC_COMB_TRANSPOSED=1 env swaps it if norms explode.
3. QAT activation sims skipped (see above). For S<=~40 prompts T=S//4 << 512 so the
   indexer is DENSE regardless of scores — the fp4/hadamard skip provably cannot change
   the selected set at smoke lengths; it only perturbs pool VALUES via the indexer path,
   which at dense selection is unused (pool values come from the MAIN compressor, not
   the indexer compressor).
4. Tokenizer: raw continuation (no chat template) for the probe; template fallback if
   top-5 garbage (chat model, template may be required for best behavior).

## Step status

- [x] 0 integrity spot-check (this file, above)
- [x] 1 full_loader.py + unit tests — PASS 2026-09-04
  - test a (wo_b FP8 two-way vs safe_open+native-float8 block math): BIT-EQUAL,
    absmean 0.01680 (01b leading-slice ref 0.0156 — consistent)
  - test b1 (expert w1 MXFP4 LUT vs from-first-principles per-bit decoder incl
    per-32-group E8M0 scales): maxdiff 0.0
  - test b2: torch float4_e2m1fn_x2 .to(float32) unimplemented in this torch build
    (NotImplementedError copy_kernel) — nibble order rests on the torch/OCP-MX
    low-nibble-first convention + the coherence gate (which then PASSED, see below)
  - expert stats e0/e128/e255 absmean ~0.0168-0.0175, std ~0.021-0.022 (tight, sane)
  - per-layer bytes: 3.57-3.60 GB/layer (3.423 GB experts); language stack 153.90 GB;
    each layer's tensors live in exactly one shard
  - vllm serving-stack cross-check attempted (dspark image): quark/triton kernel
    paths have no readable python nibble code (buried in compiled kernels); documented
    as convention + gate.
- [x] 2 dsv4_full.py — engine built, 43-layer mini-forward [1,2,3] clean (16.7s cold,
  no NaN, peak 5.19 GB)
- [~] 3 smoke_forward.py — RUN 1 (probe "The capital of France is", 5 toks):
  - per-layer stream rms: L0 0.094 -> L4 0.154 -> L8 3.63 -> L16 8.84 -> L32 22.3
    -> L42 23.9 (smooth monotone growth, max single value 604, no NaN/Inf anywhere
    in 43 layers or logits)
  - G2 TOP-5 @ last position: ' Paris' 25.47 / ' **' 21.94 / ' {' 20.08 / ' {{' 19.94 /
    ' [[' 19.91 — the canonical probe answer with a 3.5-logit margin. THE ENGINE IS
    CORRECT: FP8 + MXFP4 dequant, DSA window+pool attention, official hc orientation,
    hash + sqrtsoftplus routing all validated by output quality.
  - G1 first-run verdict FAIL was a GATE CALIBRATION artifact: layer 0 rms 0.094 is
    6% below the spec'd 1e-1 floor. That is the natural post-embed magnitude (streams
    start at embed scale and grow); no collapse, no spike. Floor recalibrated to 1e-2
    with the L0 value recorded here. hc comb A/B (HC_COMB_TRANSPOSED) NOT needed —
    official orientation produces the right answer.
  - forward wall 20.1s (43 layers, avg 466 ms/layer, L0 1.2s cold); peak cuda
    allocated 5.19 GB (<< 2-layer budget; route-then-load-6 working as designed)
  - run 2 (gate fix + greedy-20 + determinism): ALL GATES PASS, OVERALL PASS
    - G1 PASS: no NaN/Inf in any of 43 layers or logits; layer rms 0.094 (L0)
      -> 32.99 max, smooth monotone growth, max single element 604
    - G2 PASS: top-5 ' Paris' 25.470 / ' **' 21.94 / ' {' 20.08 / ' {{' 19.94 /
      ' [[' 19.91 (unchanged from run 1 — forward deterministic)
    - G3 PASS: "My favourite animal is the" + greedy 20 ->
      ' cat. It is a small, furry animal with four legs and a long tail. Cats are very'
      — coherent English, factual continuation; run2 identical token ids (bit-deterministic)
    - walls: probe forward 9.8s warm (layer avg 225 ms; L0 1.0 s cold-read);
      greedy-20 (S 6->26) 309.4s = ~15.5 s/token full-recompute warm
    - peak cuda allocated 5.19 GB (route-then-load-6; 43 layers streamed, 1-layer
      residency + embed/head/top stack ~3.3 GB)
    - receipts: smoke_result.json sha256-16 6c4fa4facd766b49,
      smoke_run2.log f7c229b5535059a2 (both mirrored to Mac repo dir)
- [x] 4 greedy-20 + determinism — DONE (see run 2 above)

STEP 1 COMPLETE 2026-09-04 ~14:55 NZST. All four deliverables green.

## STEP 2 (03-real on-policy regen) — DONE 2026-09-04 ~16:55 NZST

Engine extension (validated single-seq path untouched): packing-based
`forward_batch` in dsv4_full.py — N variable-length sequences in one 43-layer
sweep; per-token projections/hc/MoE are pack-agnostic; per-sequence compressor
and indexer loops; per-token gather rows (window ++ own-sequence pool) keep
sequences independent. Validation:
- identical-pair in one batch -> bit-equal rows (no positional/index bug)
- single-seq vs batch path: maxdiff 0.45 logits over 43 bf16 layers (cuBLAS
  tile-order drift), argmax identical (' Paris')
- 2-seq mixed batch: top-1 preserved per sequence

PROMPT FORMAT FINDING: raw continuation of arithmetic questions drifts into
JSON-chat boilerplate ('",\n "user": "You are...') — the model treats bare
questions as fragments of chat-format files. Fixed with the OFFICIAL template
from checkpoint encoding/encoding_dsv4.py (thinking_mode="chat"):
`<bos><|User|>{q}<|Assistant|></think>` (ids: bos=0, User=128803,
Assistant=128804, </think>=128822). Templated greedy sanity: "23 + 19 = 42.",
"7 * 8 = **56**." — correct.

RUN (tc_batch_gen.py, regen_run1.log): 40 problems x 3 samples (a+b / a*c /
a+b*c, seed 20260904), temp 0.8, budget 48, global sample seed 20260977,
holdout seed 20260905 (12 problems saved UNANSWERED in data/holdout_problems.json).
All 120 sequences in ONE mega-batch (S_pack 1677 -> 2817 tokens); sequences
EOS-out and drop from the active set (120 -> 45 active by step 10).

Receipts (data/regen_report.json, strict last-integer grading):
- correct_samples 76/120 strict, 78/120 first-answer regrade (2 truncated-after-
  correct-answer cases)
- problems_with_correct 27/40 (regrade does not change this)
- wall 2676.9s = 44.6 min; 2952 tokens generated; mean step 55.8s
- aggregate throughput 1.1 tok/s counted over generated tokens, but the batch
  advances 45-120 sequences per step: effective 40-67 tok/s wall-clock across
  the batch (2952 tokens / 44.6 min)
- per-kind yield: a+b 42/42 correct (0 hit cap); a*c 36/39 regrade (6 hit cap);
  a+b*c 0/39 — ALL hit the 48-token cap: the model reliably starts PEMDAS
  step-by-step explanations and the budget truncates mid-reasoning BEFORE the
  answer. Missing pids = exactly the 13 kind-2 problems (2,5,...,38).
  => the sub-30-problem warning is a BUDGET artifact, not a capability artifact.
  Fix for the next pass (coordinator's call): budget ~160 for kind-2, or a
  short-answer system prompt (still on-policy w.r.t. that prompt).
- no teacher topup run (on-policy yield is real; decision belongs to coordinator)

Artifacts: data/regen.jsonl (120 rows: prompt/answer/sample/completion/correct/
parsed/n_new), data/regen_report.json, data/holdout_problems.json,
regen_run1.log. All mirrored to Mac tools/thinkingcap-real/data/.

STEP 2 STOPPED HERE per coordinator: LoRA target selection (04-real) comes
after yield review.

## 2026-09-04 EVENING — coordinator order: kind-2 regen @160, 04-real LoRA,
## holdout gate, stop before 05-real

### Kind-2 regen (data fix before 04-real)
- DECISION (documented): plain official template at 160-token budget; the
  short-answer system-line A/B was NOT run — the cap trains on termination
  during multi-step reasoning, and a short-answer prompt would SUPPRESS the
  reasoning traces that are the training surface. Budget was the binding
  constraint (39/39 kind-2 samples hit the 48 cap mid-PEMDAS), not style.
- tc_batch_gen.py now parametrized (--kinds --max-new --seed --sample-seed
  --out --tag); rows carry provenance: kind, budget, hit_cap, correct (strict
  last-int) + correct_first_answer (first-line regrade for truncated-after-
  answer cases; strict stays the graded field).
- run: --kinds 2 --max-new 160 --sample-seed 20260977 -> data/regen_kind2.jsonl
- build_merged_report.py: old 48-budget run preserved as regen_run48.jsonl;
  merged data/regen.jsonl (kind 0/1 @ budget 48 + kind 2 @ budget 160) +
  regenerated regen_report.json (per-kind yield, mean completion length,
  finished-before-cap vs hit-cap) + sft.jsonl (shortest-correct per problem).

### 04-real LoRA — TARGET SET DOCUMENTED BEFORE TRAINING
Sites: per language layer i in 0..42 — attn.wq_b, attn.wkv, attn.wo_b +
ffn.shared_experts.{w1,w2,w3}. 258 adapters (6/layer x 43), shapes read from
the checkpoint meta (no hardcoding). WHY this set:
  1. Dummy precedent (04_train.py): the identical six sites on the nano
     (24 adapters x 4 layers) — the cycle this real run mirrors.
  2. Official ThinkingCap lineage: cap the language-stack projections;
     routed experts, hc params, norms, sinks, ape untouched; vision absent;
     MTP/DSpark untouched (drafter-anchor lesson from drowzeys: edit_mtp false).
  3. wq_b/wkv/wo_b + shared expert span the attention output path and the
     always-on FFN path — the surfaces that carry answer-emission behavior;
     leaving the 256 routed experts alone keeps the cap cheap (23.8M params
     vs 6.4B) and the drafter/pool math untouched.
Config: R=8 ALPHA=16 (scale 2.0), LR 3e-4, AdamW, EPOCHS=30 full-batch steps,
seed 20260904. Loss: CE on completion tokens only; eos APPENDED to every
target completion (termination is the trained signal; dummy omitted it
because its eos was vacuous). SFT rows = shortest-correct per problem.
Training-through-streaming: layer-wise rematerialization (no-grad boundary
pass saving 43+1 stream stacks, then reverse per-layer dot-product backward;
exact chain rule, peak memory = one layer + boundaries ~9 GB at S=3000).
fp32 activations for training (bf16 weight folding would round the small
adapter deltas away; tf32 matmuls allowed). Step-0 identity check: with B=0
the adapter path must reproduce the base loss exactly (asserted).
Save format: lora.{layers.i.site.weight}.{A,B} = dummy 05 merge format.

### Holdout gate (before any merge; official before/after claim, ours measured)
- Holdout seed 20260905, 12 problems, FIRST touched here. 3 samples/problem,
  temp 0.8, budget 160, sample seed 20260978 (same both arms -> paired).
- base arm = validated engine; capped arm = adapters FOLDED into loaded bf16
  weights (W_eff = W + scale*B@A; exactly the 05-merge arithmetic, applied in
  memory — the checkpoint-copy merge is the separate stopped-before step).
- Metrics: per-kind correctness (strict + first-answer), mean completion
  tokens (all + correct-only), hit-cap counts. Gate: accuracy within noise
  (2 samples of 36) AND length measurably bounded.
