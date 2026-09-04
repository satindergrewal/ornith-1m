# ThinkingCap dummy cycle — RUNBOOK (resumable)

De-risk the full ThinkingCap-on-DSV4 pipeline on the box for $0 before any
paid pod. Design record: DESIGN.md (approved 2026-09-01). Machine state
tracker: state.json — READ IT FIRST when resuming.

## Where things live

| thing | where |
|---|---|
| work dir (box) | /mnt/t5evo/thinkingcap-dummy/ |
| work dir (Mac, scripts only) | ~/Documents/GitHub/ornith-1m/tools/thinkingcap-dummy/ |
| nano checkpoint | box: /mnt/t5evo/thinkingcap-dummy/nano-dsv4-vision/ |
| real tensor index | box: vision-exp-index.json (72,633 names, DeepSeek-V4-Flash-Vision-Exp) |
| nano config | box + repo: nano-config.json |
| runner container | glm53-k35-tp2capture (has torch, safetensors; mount /wd) |

Standard invocation (no GPU needed for steps 01-02, 05):

```bash
ssh satinder@<box> 'docker run --rm -v /mnt/t5evo/thinkingcap-dummy:/wd -w /wd \
  --entrypoint python3 glm53-k35-tp2capture <step_script>.py'
```

NEVER put `sleep` inside an ssh session (known drop cause). Poll with short
ssh calls; do local work between polls.

## Steps

RESTRUCTURED 2026-09-03 (his call, correct): the real run is
cap-on-community-ablit (drowzeys base on box at
/mnt/t5evo/dsv4-vision-ablit, 174G, FP8, MTP-3 preserved), so the dummy
must start life already-abliterated to rehearse the real pipeline;
self-abliteration leaves the critical path.

1. **01_synth.py — DONE (2026-09-02), stock-base arm.** Maps all 72,633
   real tensor names to the nano architecture; 607 tensors, 78.96M
   params, 151 MB, seed 20260902. Coverage closed; skips are by-design
   index skips. Kept as the stock arm of the ordering A/B.
1b. **01b_synth_from_ablit.py — NEXT (mainline).** Slice the nano dummy
   OUT OF the drowzeys FP8 checkpoint: dequant the FP8 block-scale
   slices to BF16, map nano layer k <- real layer 10+k so the nano
   carries their actual wo_b L10-35 edit, vision/MTP/top slices from
   their real counterparts, real tokenizer. Deliverable: loadable nano
   that IS an ablit derivative. Bonus: rehearses the FP8 dequant
   handling the real run needs anyway.
2. **02_ablit_direction.py — OPTIONAL A/B ARM ONLY.** Self-ablit on the
   stock nano (wo_b projection, L10-35, lambda 3.5, 26 tensors,
   drowzeys method). Needed only for the cap-then-ablit vs
   cap-on-ablit ordering experiment; NOT on the critical path.
3. **03_regen_data.py — DONE (2026-09-04).** 0/100 on-policy correct
   (dummy, as designed); 100-row off-policy teacher topup, flagged.
4. **04_train.py — DONE (2026-09-04).** LoRA via nano_torch autograd,
   24 adapters / 74,240 params; loss 12.06->11.88 (flat = expected).
   **04b_eval.py — DONE (his pre-merge ask): adapters apply in-memory,
   generate clean, probe drift 4.771.**
5. **05_merge.py — DONE (2026-09-04).** 24 tensors merged into
   nano-dsv4-vision-ablit-cap; merged-vs-base probe maxdiff 4.771 ==
   adapter-applied value = bit parity. MERGE_META.json receipt.
6. **06_ab_bench.py — DONE (2026-09-05, his demand: test OUR dummy work).**
   Base (nano-dsv4-vision-ablit) vs the MERGED cap artifact
   (nano-dsv4-vision-ablit-cap), greedy, 160/256-token budgets.
   RESULTS (cap/ab_bench.json on LilMonkey): 30/30 math + 2/2 open
   prompts diverge at token 0; logits maxdiff 6.52; identical_streams 0.
   Lengths equal (all budget-capped, no EOS — the 79M slice has no
   arithmetic skill, 0/30 both arms; mechanism rehearsal, not quality).
   Base text loops immediately ("dátummal" x13); cap soup differs with
   far less concentrated repetition. THE receipt that 01b->05 produces
   a model that demonstrably differs from base.
7. **07_gates.py — REAL-RUN LANE (dummy covered its merge mechanics).
   Real run next: (a) GPU adaptation of nano_torch for the full
   drowzeys model (43L/128 experts/FP8) - layer-streamed on ONE Spark
   (128GB unified); (b) real 03 data regen on-policy; (c) LoRA;
   (d) merge into raw; (e) vLLM whole-model serve + before/after
   reasoning-token/correctness bench (official ThinkingCap method).
   Loop-check gate: stock vs capped reasoning length at equal quality.

## 2026-09-04 status: DUMMY CYCLE COMPLETE (01b->05). Real run = the
Spark lane above; hotdogs Qwen mixed-quant campaign queued AFTER it.

## Resume protocol

1. Read state.json — find the first step with status != done.
2. If the step script exists in the repo, ship it (scp) and run per the
   standard invocation. If not, write it against the DESIGN.md spec.
3. On success: update state.json (status, artifact path, receipt hash),
   then move to the next step.
4. GPU steps: check what is serving on :8012 first
   (`docker ps --format '{{.Names}}' | grep glm53-flash`), stop cleanly
   (docker rm -f + verify VRAM free), run, then restore the serve with
   the exact env from serve-boot17.log's launch line.

## Gotchas banked so far

- REAL-RUN CORRECTIONS (2026-09-04, sparky1 tc-real): the real model has 256
  experts (not 128) and experts are MXFP4 (I8 nibbles + per-32-group E8M0
  scales), NOT FP8; the checkpoint ships its OWN pure-torch reference engine
  at inference/model.py - it outranks nano_torch AND exllamav3 where they
  conflict (official hc comb is untransposed). Regeneration MUST use the
  official template `<bos><|User|>{q}<|Assistant|></think>` (raw continuation
  drifts into JSON-chat boilerplate). 48-token budgets kill multi-step
  reasoning traces (a+b*c class) - use >=160.
- STACK PIVOT (2026-09-03): transformers 5.16's deepseek_v4 class is a
  DIFFERENT parameterization from the raw checkpoint (rope rows split
  out of q_b/kv projections, compressor uses 2*head_dim direct projs
  instead of the lora-compressed 512+64 layout, hc params expanded via
  hashing, no MTP/vision at all). Raw->HF conversion (01c_to_hf_names.py)
  got 98 language tensors name-mapped but shapes cannot match without a
  lossy re-projection. 01c is therefore OPTIONAL/curiosity; the dummy's
  engine is nano_torch.py (pure torch, raw names, mirrors
  exllamav3/modules/dsv4.py math; CPU-capable; used by 03/04/05).
  exllamav3 on the 5090 (post-handover) is the independent engine for
  the 07 greedy-equivalence gate.
- 01_synth.py reads WD from env SYNTH_WD (default /wd = the container
  mount). Do NOT hardcode host paths inside container scripts.
- Shape tables are the source of truth for coverage; a falsy shape ()
  is valid (scalar scales) — test with `is not None`, never truthiness.
- Real checkpoint uses PER-EXPERT tensors (ffn.experts.{e}.wN.*), not
  fused; mtp layers carry their own per-expert stack + confidence_head.
- Indexer layers are EVEN-numbered only (2, 4, ... 42); nano keeps layer 2.
