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
3. **03_regen_data.py — TODO.** Hotdogs-method on-policy regen:
   grade-school arithmetic, 3 samples from CURRENT base, oracle filter,
   shortest-correct SFT set + short-vs-verbose DPO pairs.
4. **04_train.py — TODO.** LoRA on nano (language + expert layers; vision
   tower + nextn FROZEN — drafter-anchor lesson). Needs GPU: coordinate
   with product serve on :8012 (Satinder gave start/stop authority this
   session; restore after).
5. **05_merge.py — TODO.** LoRA -> merged BF16 checkpoint.
6. **06_reencode.py — TODO.** ModelOpt FP8/NVFP4-experts quant on the 6000s
   (the BF16-LMHead fix lane proved this toolchain here). GPU step.
7. **07_gates.py — TODO.** Bake-serve merged nano on the day-0 vLLM image;
   greedy-equivalence, DSpark acceptance through the 3-layer nextn,
   loop_eval, stock-vs-knob-vs-cap 3-arm. GPU step.

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

- 01_synth.py reads WD from env SYNTH_WD (default /wd = the container
  mount). Do NOT hardcode host paths inside container scripts.
- Shape tables are the source of truth for coverage; a falsy shape ()
  is valid (scalar scales) — test with `is not None`, never truthiness.
- Real checkpoint uses PER-EXPERT tensors (ffn.experts.{e}.wN.*), not
  fused; mtp layers carry their own per-expert stack + confidence_head.
- Indexer layers are EVEN-numbered only (2, 4, ... 42); nano keeps layer 2.
