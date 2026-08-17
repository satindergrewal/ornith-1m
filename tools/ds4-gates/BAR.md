# DS4 bar

Phrase: **ds4 bar**
When Satinder says that, read this file and answer met / not met. Do not narrate history.

## Goal (met only if ALL are true)

On a daily model (DSV4, Inkling, Ornith/Qwen3.5, GLM, or the Aug 7 short list), Metal, `-np 1 --kv-paged`:

1. One 256k–1M master stays resident in a single shared KV pool.
2. Child agents admit against the **pool**, not against a slot count.
3. Children share the master's prefix by refcount (not a copied slab).
4. If the pool is full: queue or swap. Never reject-because-slots. Never slice 1M into 4×256k.
5. A child leaving does not copy or destroy the master.

Decode may be single-width. Speed ≥ static is a later gate, not this one.

## Current verdict

**NOT MET.** 2026-08-17 13:09 GST

Qwen3.8 27B only until Satinder says otherwise. DSV4 parked. Official item 1 is still a 256k–1M resident master.

Live (not MET):
- 1M fill **aborted** 2026-08-17 18:29 GST (Satinder). PID 20448 SIGINT. Port 18140 down.
- Last progress ~347k / 1048320 (~33%). Hello 200, 1M prompt admitted, no crash. 1M inherit **unverified**. He will test `/fork` in real use.
- 20/50/100 spray skipped. 8k/16k inherit still holds. No MET.


Landed on `ds4-ports` (not pushed to fleet):
- `4af52cc4c` Cut 1: `-np` is batch width; bookkeeping grows
- `9e0d9f165` fitter = one master + headroom, not `n_ctx × n_parallel`
- `fc940c2e7` pool-full: preempt unref tails, waiters not 500'd
- `564245b6d` finished master prefix stays parked; later children admit SHARED
- `54996eea2` session_id / POST /fork / POST /close_session
- `5cb7a0a69` unknown session_id fails loud
- `471672ff0` named session survives prefix shorter than one block
- `9f12eb160` kv_paged refuses n_gpu_blocks==0 instead of GGML_ASSERT abort
- `d11c037f3` child named /fork reports inherited tokens as HTTP cache_n (was always 0 on paged)
- `6911a61ac` paged named /fork does not bill inherited tokens as prompt_n
- `257458bdd` named/held master prefix is not an eviction victim
- `a7359a75f` named master filling GPU: children WAIT (not CPU swap, not 500) until /close_session
- `2474fd5fb` overlap crash: defer second hybrid child past n_seq_max
- `06fba44e2` --kv-paged default-on for Qwen35/QWEN35MOE only
- `dfd4800f6` Qwen hybrid RS 4 cells / 598.50 MiB (live log). 8k overlap /fork child1 cache_n=128 id_slot=0; child2 cache_n=128 id_slot=1; unknown 400; sequential reuse own JSON 128/128. Still 3-live+hold wall.
- `d5ffab3aa` named-hold unique suffix GPU→CPU (3 blocks); stories15M 8k child 200 without /close_session, cache_n=8000, n_past stayed 8048. Mixed-table child still blocked.
- `cd0b32cda` fail mixed-remap child once (no ERROR spin). wm2 8k child still 500 outgrew pool. Log 5946 bytes, fail_once=1.
- `cd0b32cda` wm8 prefix-full child 200: cache_n=7984 prompt_n=28, master n_past 8000, mixed-scratch used, no fail-once. stories15M 8k. Prefix GPU ids not logged.
- `a486fd46f` named-session grow must not abort as zero-remainder prefill. Qwen 8k grow 200 tokens_cached=8048. Swap not proven.
- `dd2925b6e` swap parked unique suffix before fail_mixed. Qwen 8k @512 child 200 cache_n=8000, DS4P-SWAP 7 blocks, session not shortened.
- `1be7e557b` pin named/held prefix so prompt-cache LRU cannot drop it
- `2411f4b44` pin rebind (tree; 8q6 binary was 1be7e557b)
- `72bac86ca` paged hybrid RS cells 4 → 24 (20 mid-work occupancy; not 256)
- `b28507227` reserve waiter unique on CPU at enqueue (-np 1; no GPU steal)
- `dcdb77768` DS4P-QUEUE ERROR on enqueue-short (visible at -lv 2)
- `5eb1bfbea` paged hybrid RS cells 24 → 40 (20–40 mid-work occupancy; hold-cell kept; not 256)
- `fa3fa8527` named session after 200 is held, not a live runner for waiter CPU reserve
- `6314227da` hybrid RS unit retarget 24 → 40 (n>40 still returns n)
- `099fc807f` named grow keeps unique GPU suffix on the hold (leftover stays short)
- `3f6a54c01` revert too-strict named-grow unit assert
- `afded1229` eviction-victim hold N includes unique suffix
- `f733080a6` sweep remaining named-hold N==Ku to include unique suffix
- `68a0c213c` accept paged `-b 512 -ub 256`; bill KV-holding layers; drop 1.50 pool headroom
- `d4ef6e27d` bind hybrid full-attn layers to paged mctx (stop silent STATIC 1M child)
- `ccdf090fa` init_full dummy host block_table/write_slots (bs<=32 set_input SIGSEGV)
- `440817162` size paged pool so watermark + decode still leave one n_ctx master


Still missing for MET:
- Qwen3.8 27B 1M: fill aborted at ~347k/1048320 (~33%). Admitted, no crash. Inherit **unverified**. Real-use `/fork` when he wants. 20/50/100 spray skipped. No MET.
- Metal batched decode (several sequences in one graph). Not `-np`. Not CUDA. Parked until the 1M `/fork` proof. Mac GPU only.
- Radix auto-share (SGLang-style) later, on this same pool. `/fork` of a named master is the explicit share for now.
- DSV4 Flash 0731 8k: named /fork HTTP proven on new binary HEAD a7359a75f (child cache_n=64 prompt_n=22 tokens_evaluated=86; unknown 400; n_gpu_blocks=192 no overcommit). Parallel /fork PASS on ab2507c5e (both inherit 128/139). Decode still single-width. Parked. Not 256k.
- DSV4 Flash 0731 8k two-child named /fork: both cache_n=128; overlap deferred at n_seq_max=1, no crash; master still forkable; unknown 400. JSON in llama.cpp-ds4ports/.scratch-dsv4-8k-two-child/. HONEST: queue-then-serial, not concurrent usable serve. Sink layers still static. 8k is not 256k.
- Qwen3.8 27B 8k: named /fork HTTP proven on HEAD a7359a75f (child cache_n=128 prompt_n=12 tokens_evaluated=140; child2 128/16/144; unknown 400; DS4P_PAGED_HYBRID=1, n_gpu_blocks=768, all attention layers paged path). Hybrid still builds attn_kv_size=n_ctx_seq first (not slab-gone). Decode gate pending. Default-on for Qwen35/QWEN35MOE as of 06fba44e2 (8k /fork env unset: cache_n=128 prompt_n=12 tokens_evaluated=140). Other hybrids still opt-in (silent-static risk). Not 256k.
- Qwen3.8 27B `-c 32768`: named /fork HTTP started on HEAD a7359a75f (child cache_n=32 prompt_n=6 tokens_evaluated=38; unknown 400; n_gpu_blocks=3072 no overcommit; RSS ~22 GiB). HONEST: 32-token prompt is not a 32k-context proof. Not 256k.
- Qwen3.8 27B two children named /fork from one 16k master: both cache_n=16384; overlap crash fixed 2474fd5fb (defer second hybrid child past n_seq_max); master still forkable after; unknown 400. JSON in llama.cpp-ds4ports/.scratch-two-child-16k/. HONEST: serial/defer at -np 1, not a concurrent usable serve. Not 256k.
- Qwen hybrid RS: `dfd4800f6` re-proof log kept (`.scratch-8k-qwen-reproof/server.log` 89785 bytes): `size = 598.50 MiB (4 cells, 64 layers, 1 seqs 0 rs_seq)`; froze hold seq 3; grew slot 1. Overlap JSON child cache_n=128/128 unknown 400 wall-overlap true. Sequential own JSON 128/128 same slot. Still a 3-live+hold slot wall, not pool admission. 20–40 later. 8k is not 256k.
- Qwen3.8 27B `-c 32768` 28k-prefix named /fork: master 28672 tokens / 306.4s; child cache_n=28672 prompt_n=9 tokens_evaluated=28681 in 1.66s; 1792 blocks by reference; unknown 400. JSON in llama.cpp-ds4ports/.scratch-28k-named-fork/. HONEST: 28k is not a 32k fill, not 256k.
- Qwen3.8 27B `-c 32768` 24k-prefix named /fork: master 24576 tokens / 241.6s; child cache_n=24576 prompt_n=9 tokens_evaluated=24585 in 1.25s; 1536 blocks by reference; unknown 400. JSON in llama.cpp-ds4ports/.scratch-24k-named-fork/. HONEST: 24k is not a 32k fill, not 256k.
- Qwen3.8 27B `-c 32768` 16k-prefix named /fork: master 16384 tokens / 135.2s; child cache_n=16384 prompt_n=9 tokens_evaluated=16393 in 1.04s; 1024 blocks by reference; unknown 400. JSON in llama.cpp-ds4ports/.scratch-16k-named-fork/. HONEST: 16k is not a 32k fill, not 256k.
- Qwen3.8 27B `-c 32768` long-prefix named /fork: master 9241 tokens / 63.8s; child cache_n=9232 prompt_n=21 tokens_evaluated=9253 in 0.92s; 577 blocks by reference; unknown 400. JSON in ornith-1m/_scratch/qwen38-fork-32k-long-18091/. HONEST: 9k is not a 32k fill, not 256k.
- Swap is partial. Qwen 8k/16k SWAP holds. DSV4 8k SWAP now holds on tight128 (`f733080a6` PID 46641): leftover 2, DS4P-SWAP 1 block, session not shortened, 3/3 cache_n=8000. QUEUE not yet on DSV4. Tight mixed-table still parked. 8k/16k are not 256k.
- concurrent agents on a daily model as a usable serve (not a one-shot e2e)

## Dated notes

### 2026-08-17 — /fork-required client is the wrong product

- Satinder: harness talks stock chat/completions; multi-agent share is automatic at the API. Not Spock headers, not `/fork` from CC/OpenAI/Grok-Build.
- Next cut: radix/auto prefix-share on `/v1/chat/completions`. `/fork` optional.
- 28943 left up on 18140 for real use. No MET.


### 2026-08-17 — 1M fill aborted; inherit unverified

- Satinder stopped verification. PID 20448 SIGINT. Last ~347k/1048320 (~33%), ~12 tok/s, ~8h in.
- Paged 1M launch + admit + Hello 200 stood. No 1M master parked. No 1M `/fork`.
- 20/50/100 spray skipped. He will test in real use and report if it breaks.
- 8k/16k inherit + QUEUE + leave + SWAP still hold. No MET.


### 2026-08-17 — Qwen-only 1M shared-pool plan (Satinder)

- Model: Qwen3.8 27B only. DSV4 parked.
- Shape: match `ornith-models/Qwen3.8-27B-GGUF/run.sh` (Q4 or Q8 weights, **q8 KV**, `-c 1048576` YaRN + override-kv, `-b 512 -ub 256`, `-fa on --fit off`). Paged. Do not claim 1M cannot fit. Static already does.
- Now: fill named 1M master, then `/fork` 20/50/100. Inherit + leave-master-up = HTTP shared-pool setup for this model. Decode still one-wide. Not speed-vs-static. Not Spock-wired. No MET until those forks land.
- After that: Metal paged decode that batches several sequences in one graph. Box/CUDA is not that kernel. Do not raise `-np`.
- Later: radix auto-share on this pool (SGLang-style). `/fork` stays the explicit path.
- Not NIAH. Not 8k/16k theater as done.
- Earlier 262k/1M "doesn't fit" / Metal OOM was our miss: f16 KV, 1.50 headroom, full-attn STATIC double-bill, then watermark 0.05 making usable ~996k. Not the Mac.
- Live PID 20448 HEAD `440817162` pool 34502. Fill running. No MET.

### 2026-08-17 — Qwen 1M Hello 200, fill 500 capability contract

- HEAD `d4ef6e27d` PID 84028. `.scratch-qwen-1m-q8w-f16kv-paged-attn/`
- Hello 200. Full-attn bind landed. Fill 1048320 HTTP 500. `bs=64 × head_dim 256 > 8192` without champion. Layers still static-path.
- Did not flip DS4P_METAL_CHAMP. Next: --kv-block-size 16 (16×256=4096). Same 1M. No MET.

### 2026-08-17 — Qwen 1M Q4 f16 paged Metal OOM

- HEAD `68a0c213c` PID 65225. `.scratch-qwen-1m-q8w-f16kv/`
- 512/256 accepted. Fitter 16384 blocks 1.0×. n_ctx=1048576 health 200.
- Smoke Hello 500. Metal OOM. Full-attn layers 3,7,…,63 `took the STATIC path -- no paged context`.
- Double KV: paged pool + static 1M on those layers. Agents not run. No MET.

### 2026-08-17 — Qwen3.8 27B native 262k FAIL at fill

- HEAD `f733080a6`. `.scratch-qwen-262k-agents/`
- Auto-fit: 262144 needs 6144 KV blocks / 96.0 GiB; budget 4847 / 75.7 GiB. Largest n_ctx ~206784.
- Retry `--n-gpu-blocks 4096` fill HTTP 500 deadlock (prefix 261450). Agents not run.
- HONEST: that 96 GiB bill was f16 KV + 1.50 headroom, not "Mac cannot hold 262k". Static 1M q8 KV already fits. No MET.

### 2026-08-17 — DSV4 8k leftover-short QUEUE inherit yes overflow no

- HEAD `f733080a6` PID 46641. `.scratch-dsv4-8k-tight128-queue/`
- Harness OK. 9/9 HTTP 200 cache_n=8000. leftover at spray=1. unique 160 tok = 3 blocks.
- enqueue-cpu-unique=0 QUEUE=0 SWAP=0. Waiter unique only after holder RELEASE (leftover recovered to 3).
- Product hole: leftover 1 < unique 3 at POST and waiters still did not CPU-reserve. No more HTTP spray. No MET.

### 2026-08-17 — DSV4 8k tight128 leave-safe after SWAP

- HEAD `f733080a6` PID 46641. `.scratch-dsv4-8k-tight128-leave/`
- One /fork HTTP 200 cache_n=8000 prompt_n=9. SWAP=0 (suffix already CPU). Master still named.
- DSV4 8k inherit + SWAP + leave-safe closed on this HEAD. QUEUE not yet. No MET.
- Next: leftover-short fat unique on same PID (QUEUE), not another small /fork.

### 2026-08-17 — DSV4 8k tight128 SWAP

- HEAD `f733080a6` PID 46641. `.scratch-dsv4-8k-tight128/`
- 7002 SIGINT. `--n-gpu-blocks 128 --n-cpu-blocks 192` wm0. Fill 8008 leftover 2.
- 3/3 HTTP 200 cache_n=8000. SWAP 1 block GPU→CPU, n_past=8008, session not shortened.
- QUEUE=0. enqueue-cpu-unique=0. Occupancy 1. No MET.
- Next: one /fork after SWAP so leave-safe still holds with unique on CPU.

### 2026-08-17 — DSV4 8k leftover-starve grow still plenty

- HEAD `f733080a6` PID 7002. `.scratch-dsv4-8k-starve-head/`
- Named grow 184 tok (ctx room). leftover 62→59. Need ~3776 tok to hit leftover 1–4. Did not raise -c.
- No MET. Next: SIGINT 7002, relaunch same binary with tighter GPU (`--n-gpu-blocks 128`), not a ctx raise.

### 2026-08-17 — DSV4 8k sequential 2hold leftover-plenty

- HEAD `f733080a6` PID 7002. `.scratch-dsv4-8k-2hold-seq/`
- Harness OK: hold01 first-byte then hold02+2 children while hold01 live. 4/4 HTTP 200 cache_n=8000.
- leftover 64→62 stayed plenty. enqueue-cpu-unique=0. Waiter unique only after hold01 RELEASE. Occupancy 1.
- No MET. Next: named grow to starve leftover, then holder+children.

### 2026-08-17 — DSV4 8k 2hold harness miss

- HEAD `f733080a6` PID 7002. `.scratch-dsv4-8k-2hold-head/`
- Both holders POSTed same instant. hold02 finished before hold01 first-byte. Children never posted. enqueue-cpu-unique=0. Unique did not stack.
- Harness miss, not a product fail. No MET. Next: sequential first-byte then second hold + 2 children.

### 2026-08-17 — DSV4 8k four-child inherit (serial)

- HEAD `f733080a6` PID 7002. `.scratch-dsv4-8k-four-child-head/`
- 4/4 HTTP 200 cache_n=8000. id_slots all 0. Peak extra cell 1 + named hold = 2. Sequential reuse, not 4 overlapping decodes.
- No MET. Next: hold-open 2–4 overlapping unique cells on same PID. No -np raise.

### 2026-08-17 — DSV4 8k leave-safe on current HEAD

- HEAD `f733080a6` PID 7002. `.scratch-dsv4-8k-two-child-head-leave/`
- 2/2 HTTP 200 cache_n=8000. session-not-found=0. Master still named.
- No MET. Next: 4-way occupancy (2–4 live cells, not 40, not 256).

### 2026-08-17 — DSV4 8k two-child inherit on current HEAD

- HEAD `f733080a6` PID 7002. `.scratch-dsv4-8k-two-child-head/`
- Master 8008 in 40.4s (real 8k, not 128-token toy). child1/2 HTTP 200 cache_n=8000. unknown 400.
- No --n-gpu-blocks. No MET. Next: 2-way leave-safe on same PID.

### 2026-08-17 — test-paged-kv ALL PASSED after hold-size sweep

- HEAD `f733080a6` PID 28293. CPU test 4.57MB. ctest 100% 0.84s.
- 8k/16k inherit + QUEUE + leave + SWAP + grow-hold unique + unit suite closed at allowed sizes. No MET.
- Parked: 256k–1M, speed-vs-static, mixed-table Metal, concurrent decode. Do not spray 8k/16k HTTP.

### 2026-08-17 — test-paged-kv stale 4-block hold assert

- HEAD `3f6a54c01` PID 28293 left UP. CPU rebuild.
- grow-finish tests OK. `test_named_master_not_eviction_victim` expected N==4u, hold is 5 (unique suffix kept).
- No MET. Next: retarget that assert to "not shortened", including unique suffix.

### 2026-08-17 — 16k swap3-leave master still forks after SWAP

- HEAD `3f6a54c01` (product `099fc807f`) PID 28293. `.scratch-16k-qwen-overlap-1024-swap3-leave/`
- One /fork HTTP 200 cache_n=16000 prompt_n=129. SWAP=0 (suffix already CPU). session-not-found=0.
- 16k inherit + QUEUE + leave + SWAP + post-SWAP fork closed on this HEAD. No MET.
- Next: rebuild test-paged-kv CPU and ALL PASSED. 28293 stays UP.

### 2026-08-17 — 16k SWAP on named-grow hold

- HEAD `099fc807f` PID 28293. `.scratch-16k-qwen-overlap-1024-swap3-head/`
- Rematch 16001. Grow n=15 free_before=23 leftover 23→8. Child 200 cache_n=16000.
- DS4P-SWAP 1 block (n_past=16001) then 15 blocks (n_past=16225). Session not shortened.
- No MET. Next: one /fork after SWAP so leave-safe still holds with unique on CPU.

### 2026-08-17 — 16k SWAP fail: grow-finish returns leftover

- HEAD `6314227da` PID 79509. swap2 child 200 cache_n=16000. SWAP=0.
- Grow leftover 14→1, then grow-finish recovered. Child CHECKOUT n=11 free_before=12. RELEASE n=10 immediately before.
- Same hole as swap-head. Not a harness miss. Named grow unique is not staying on the hold. No MET.

### 2026-08-17 — 16k SWAP FAIL leftover recovered to 11

- HEAD `6314227da` PID 79509. `.scratch-16k-qwen-overlap-1024-swap-head/`
- Grow GPU leftover n=14 free_before=16 leftover 16→2. tokens_cached=16224 (parent+160=n_ctx).
- Then LRU prompt-cache evict 1167 MiB, RELEASE n=1014 and n=8. Child CHECKOUT n=11 free_before=11. SWAP=0. Inherit 200 cache_n=16000.
- Cannot grow more on this session. No MET. Next: close, rematch 16000, grow ALL leftover (leftover 0), then one 160 /fork.

### 2026-08-17 — 16k leave-safe on current HEAD

- HEAD `6314227da` PID 79509. `.scratch-16k-qwen-overlap-1024-leave-head/`
- All 8 HTTP 200 cache_n=16000. session-not-found=0. Master survived QUEUE overflow.
- 16k inherit + QUEUE + leave closed on this HEAD. No MET. Next: 16k unique-suffix SWAP (never fired on 16k).

### 2026-08-17 — 16k leftover-starve QUEUE on current HEAD

- HEAD `6314227da` PID 79509. `.scratch-16k-qwen-overlap-1024-2h21h-head/`
- Holder in-flight. 22/22 HTTP 200 cache_n=16000. QUEUE=5 ERROR req 18–22. enqueue-cpu-unique=17 last free_before=16.
- Matches `dcdb77768` 16k QUEUE. No MET. Next: 8-way leave-safe on same PID.

### 2026-08-17 — 16k inherit 8/8 on current HEAD

- HEAD `6314227da` PID 79509. `.scratch-16k-qwen-overlap-1024-8-head/`
- Master 16001 in 130.4s (real 16k fill). All 8 HTTP 200 cache_n=16000 in 19.95s. session-not-found=0.
- No MET. Next: leftover-starve QUEUE on same PID (last 16k QUEUE was `dcdb77768`).

### 2026-08-16 — test-paged-kv ALL PASSED at 40 cells

- HEAD `6314227da` PID 84486. CPU test binary 4.6MB. `test_named_grow_after_finish_skips_cpu_reserve` ran OK.
- 8k/16k inherit + QUEUE + leave + 40 RS + SWAP + swap2 race are closed at allowed sizes. No MET.
- Parked: 256k–1M, speed-vs-static, mixed-table Metal, concurrent decode. Do not spray 8k HTTP.

### 2026-08-16 — test-paged-kv stale 24-cell assert

- HEAD `fa3fa8527` PID 84486 left UP. CPU rebuild only.
- `test_hybrid_rs_few_live_cells` expected 24, got 40 (`LLAMA_HYBRID_RS_CELLS_PAGED` from `5eb1bfbea`). New named-grow test never ran.
- No MET. Next: retarget that test to the 40-cell cap, rerun ALL PASSED.

### 2026-08-16 — 40rs-swap4 immediate grow uses GPU leftover

- HEAD `fa3fa8527` PID 84486. `.scratch-8k-qwen-overlap-512-40rs-swap4/`
- 33387 SIGINT 22:23 GST. Same 8k flags. grow POST 27ms after master 200, no idle wait.
- grow CHECKOUT n=12 free_before=12 leftover 12→0. enqueue-cpu-unique=0.
- Child HTTP 200 cache_n=8000. DS4P-SWAP 11 blocks GPU→CPU, n_past=8176, session not shortened.
- No MET. Next: rebuild test-paged-kv (CPU) and run the new test. 84486 stays UP.

### 2026-08-16 — 40rs-swap3-leave master still forks after SWAP

- HEAD `5eb1bfbea` PID 33387. `.scratch-8k-qwen-overlap-512-40rs-swap3-leave/`
- One /fork HTTP 200 cache_n=8000 prompt_n=129. SWAP=0 (suffix already CPU). session-not-found=0.
- 8k/16k inherit + QUEUE + leave + 40-cell occupancy + unique SWAP are done at allowed sizes. No MET.
- Next: swap2 race. Immediate named grow after master 200 must not enqueue-cpu-unique.

### 2026-08-16 — 40rs-swap3 unique suffix SWAP on current HEAD

- HEAD `5eb1bfbea` PID 33387. `.scratch-8k-qwen-overlap-512-40rs-swap3/`
- Idle after rematch, grow CHECKOUT n=12 free_before=12 (GPU leftover). leftover 12→0.
- Child HTTP 200 cache_n=8000 prompt_n=160. DS4P-SWAP 11 blocks GPU→CPU, session not shortened.
- swap2 CPU path was the race. No code hole. No MET.
- Next: one /fork after SWAP (master unique now on CPU) so leave-safe still holds.

### 2026-08-16 — 40rs-swap2 grow went CPU, child not posted

- HEAD `5eb1bfbea` PID 33387. `.scratch-8k-qwen-overlap-512-40rs-swap2/`
- close 200. rematch master 8001 in 0.531s. grow 8177 via enqueue-cpu-unique n=12 free_before=192. SWAP=0.
- Cause: queue_request reserves CPU unique when running/waiting non-empty. Grow posted while new master still running.
- No MET. Next: close, rematch 8000, wait idle, grow onto GPU leftover, then one fat /fork.

### 2026-08-16 — 40rs-swap1 inherit yes, SWAP no

- HEAD `5eb1bfbea` PID 33387. `.scratch-8k-qwen-overlap-512-40rs-swap1/`
- One /fork HTTP 200 cache_n=8000 prompt_n=160. SWAP=0 QUEUE=0 enqueue-cpu-unique=0.
- CHECKOUT n=11 free_before=12. Parked master has no unref GPU suffix (leftover 12). Child unique cannot go 13+ blocks without n_prompt>8192.
- No MET. Next: close old master, new 8000 fill, grow unique onto GPU, then one fat /fork so leftover < unique.

### 2026-08-16 — 40rs leave-safe master survived overflow

- HEAD `5eb1bfbea` PID 33387. `.scratch-8k-qwen-overlap-512-40rs-leave/`
- All 8 HTTP 200 cache_n=8000 in 14.48s. session-not-found=0. QUEUE=0 SWAP=0.
- 8k/16k inherit + QUEUE + leave-safe + 40-cell occupancy are done at allowed sizes. No MET.
- Next: SWAP as overflow (bar item 4), not another occupancy spray.

### 2026-08-16 — 40-cell RS occupancy past 23-live wall

- HEAD `5eb1bfbea` PID 33387. `.scratch-8k-qwen-overlap-512-40rs/`
- Master `qwen38-8k-overlap-512-40rs-master` cached 8001, grow 8176.
- All 40 HTTP 200 cache_n=8000 in 74.4s. Occupancy id_slot 0..38 (39 unique, last cell named hold). child40 finished first on slot 0; child23 reused slot 0 last. Not a 23-live wall.
- this-run QUEUE=18 SWAP=0 session-not-found=0 fail_mixed=0 enqueue-cpu-unique=22.
- No MET. Next: 8-way leave-safe on same PID (master still /fork after 40-way overflow).

### 2026-08-16 — 16k leave-safe master survived overflow

- HEAD `dcdb77768` PID 7780. `.scratch-16k-qwen-overlap-1024-leave/`
- All 8 HTTP 200 cache_n=16000 in 19.5s. session-not-found=0.
- 8k and 16k inherit + QUEUE + leave-safe are done. No MET. Next RS 24→40, then 40-way occupancy.

### 2026-08-16 — 16k leftover-starve QUEUE proved

- HEAD `dcdb77768` PID 7780. `.scratch-16k-qwen-overlap-1024-2h21h/`
- Holder in-flight. All 22 HTTP 200 cache_n=16000 prompt_n=160. wall 77.7s.
- QUEUE=5 ERROR (req 18–22). enqueue-cpu-unique=17 last free_before=16.
- Overflow proved at 16k. No MET. Next: master still /fork after 16k overflow.

### 2026-08-16 — 16k Qwen 8/8 inherit

- HEAD `dcdb77768` PID 7780. `.scratch-16k-qwen-overlap-1024-8/`
- -c 16384 --n-gpu-blocks 1024. Master 16001 in 129.5s. All 8 HTTP 200 cache_n=16000 in 19.9s.
- QUEUE=0 (8 small unique). No MET. Next 16k leftover-starve QUEUE (holder+22×160).

### 2026-08-16 — 2h21h-leave master survived overflow

- HEAD `dcdb77768` PID 90533. `.scratch-8k-qwen-overlap-512-2h21h-leave/`
- All 8 HTTP 200 cache_n=8000 in 14.7s. session-not-found=0.
- Overflow did not destroy the named master. No MET. Next 16k inherit.

### 2026-08-16 — 2h21h QUEUE proved, 22/22 wait-then-200

- HEAD `dcdb77768` PID 90533. `.scratch-8k-qwen-overlap-512-2h21h/`
- Holder in-flight. All 22 HTTP 200 cache_n=8000 prompt_n=160. wall 46s.
- QUEUE=6 ERROR (req 17–22 leftover short). enqueue-cpu-unique=17 last free_before=16. SWAP=0.
- Overflow proved. No MET (8k, decode serial). Next: master still /fork after overflow.

### 2026-08-16 — 2h21g 22/22, 17×11 then leftover 5

- HEAD `b28507227` PID 17905. `.scratch-8k-qwen-overlap-512-2h21g/`
- Holder in-flight. All 22 HTTP 200 cache_n=8000 n_prompt=8160. enqueue-cpu-unique=17 n=11 last free_before=16.
- 18th+ never reserved; later allocate after RELEASE. DS4P-QUEUE is INFO, -lv 2 drops it.
- No MET. Next: log enqueue-short QUEUE as ERROR, then 2h21h same 160-tok 22-way.

### 2026-08-16 — 2h21f 22/22 inherit, 21 reserves still fit

- HEAD `b28507227` PID 17905. `.scratch-8k-qwen-overlap-512-2h21f/`
- Holder in-flight 4.29s at spray. All 22 HTTP 200 cache_n=8000. n_prompt 8129/8137.
- enqueue-cpu-unique=21 n=9 last free_before=12. QUEUE=0. Holder was GPU leftover, CPU started 192.
- No MET. Next 2h21g: 10-block unique (~160 tok) so 21×10>192. Same one-daemon harness.

### 2026-08-16 — 2h21e watch blocked, holder finished first

- HEAD `b28507227` PID 17905. `.scratch-8k-qwen-overlap-512-2h21e/`
- Auto-review refused watch_and_spray.py. Holder GPU allocate n=12, returned 1.80s. 22 not posted.
- Unique tails were the 129/137 class. Not n_ctx.
- No MET. Next 2h21f: one daemon posts holder then 22 with no second process.

### 2026-08-16 — 2h21d 22×400 exceed n_ctx

- HEAD `b28507227` PID 17905. `.scratch-8k-qwen-overlap-512-2h21d/`
- Holder in-flight at spray. Unique suffixes too long (8208–8226 > 8192). All 22 HTTP 400.
- Never reached enqueue-cpu-unique n=9. QUEUE=0.
- No MET. Next 2h21e: same 22-way, 129/137 tails only.

### 2026-08-16 — 2h21c watch missed CPU holder

- HEAD `b28507227` PID 17905. `.scratch-8k-qwen-overlap-512-2h21c/`
- Holder enqueue-cpu-unique n=12 free_before=192. No GPU allocate. Spray never armed.
- LRU removed 667.129 MiB then RELEASE 12/500. Probe /fork master still 200.
- No MET. Next 2h21d: spray on first CHECKOUT of any kind, then 22 fat.

### 2026-08-16 — 2h21b holder on GPU, 21 waiters still fit

- HEAD `b28507227` PID 17905. `.scratch-8k-qwen-overlap-512-2h21b/`
- Holder in-flight at spray 1.7s. Runner took GPU leftover 12, not CPU.
- 21 burst 200 cache_n=8000. enqueue-cpu-unique=21 n=9 from free_before=192. Last 12≥9. QUEUE=0.
- No MET. Next 2h21c: GPU runner + 22 fat waiters. 22×9=198>192.

### 2026-08-16 — 2h21 holders released before spray

- HEAD `b28507227` PID 17905. `.scratch-8k-qwen-overlap-512-2h21/`
- hold02 returned 20:43:17.376, spray 20:43:17.470. Hold reserve gone.
- 21 burst 200 cache_n=8000. Last waiter n=9 free_before=9. QUEUE=0.
- No MET. Next 2h21b: one holder still reserved, then 21 fat. 12+21×9>192.

### 2026-08-16 — 23q-cpu 23/23, last waiter still fit

- HEAD `b28507227` PID 17905. `.scratch-8k-qwen-overlap-512-23q-cpu/`
- All 23 HTTP 200 cache_n=8000 in 42.6s. enqueue-cpu-unique=21 all n=9.
- First free_before=192, last waiter free_before=12≥9. 21×9=189. QUEUE=0.
- No MET. Next 2hold+21 burst: 2 live reserves + 21×9 > 192 CPU.

### 2026-08-16 — 2holdd waiter unique reserved before RELEASE

- HEAD `b28507227` PID 17905. `.scratch-8k-qwen-overlap-512-2holdd/`
- enqueue-cpu-unique n=8 free_before=183 request=1 before RELEASE n=503. Runner kept GPU leftover 12.
- Burst 8/8 HTTP 200 cache_n=8000. QUEUE=0 (183 CPU leftover held 10 reserves).
- Champion kept. No MET. Next 23 fat /fork on 17905: 22 waiters × 9 > 183 CPU.

### 2026-08-16 — 2holdc waiters do not reserve unique

- HEAD `72bac86ca` PID 68987. `.scratch-8k-qwen-overlap-512-2holdc/`
- Daemon held both sockets. hold01 first-byte 20:23:49, returned 20:23:53. hold02 first-byte 20:23:55.
- hold02 socket live during hold01 decode. No unique CHECKOUT for waiter until hold01 RELEASE, free_before=12.
- -np 1 champion: do not promote waiters once decode cap is full (scheduler step). QUEUE never reached.
- Burst not posted. No MET. Next: reserve waiter unique on CPU at enqueue.

### 2026-08-16 — 2holdb ExternalShell reaped holders

- HEAD `72bac86ca` PID 68987. `.scratch-8k-qwen-overlap-512-2holdb/`
- Both POSTed 20:19:16.382. Tool reaped curls at 5.4s. hold01 never first-byte while hold02 live.
- Server aborted hold02, then hold01. Burst not posted. QUEUE=0 SWAP=0.
- No MET. Next 2holdc: daemonize both streams so the tool cannot reap them.

### 2026-08-16 — 2hold missed overlap (harness)

- HEAD `72bac86ca` PID 68987. `.scratch-8k-qwen-overlap-512-2hold/`
- hold01 20:15:17–23, hold02 20:15:28–34. 4s gap. Both-open window empty.
- hold01 grow: CHECKOUT n=1 free_before 12→4→3→2→1. Unique does stack on one stream.
- Burst 8/8 200 after holders returned. QUEUE=0 SWAP=0.
- No MET. Next 2holdb: both holders POST in the same instant, burst while both open.

### 2026-08-16 — 23q 23/23 inherit, unique never stacked

- HEAD `72bac86ca` PID 68987. `.scratch-8k-qwen-overlap-512-23q/`
- All 23 HTTP 200 cache_n=8000 in 41.4s. id_slot 0..22 occupancy 23.
- SWAP=0 QUEUE=0. Pattern: RELEASE n=509 + n=8, CHECKOUT n=9 free_before=12. Serial unique.
- RELEASE 509 is child refcount drop, not master GPU eviction (next inherit 8129/8129 at 0s).
- No MET. Next hold-open: 2 stream /fork stay in-flight, then 8 n_predict=8. Unique must stack.

### 2026-08-16 — 20rs occupancy 20 > 3

- HEAD `72bac86ca` PID 68987. `.scratch-8k-qwen-overlap-512-20rs/`
- LLAMA_HYBRID_RS_CELLS_PAGED=24. Same 512 flags. No --cache-ram.
- Master 8001, grow 8176. All 20 HTTP 200 cache_n=8000 in 37.5s. id_slot 0..19. /slots peak 20.
- SWAP=2 (grow). QUEUE=0. 20×9 unique still fits 192 CPU leftover.
- No MET. Next 23q leftover-starve on 68987 (23 live + hold = 24; 23×9 > 192).

### 2026-08-16 — 40q6 40/40 inherit, overflow still dark

- Binary `1be7e557b` PID 24475. `.scratch-8k-qwen-overlap-512-40q6/`
- All 40 HTTP 200 cache_n=8000 in 73.4s. Fat 137 on 10 children. No 400/500/hang.
- SWAP=0 QUEUE=0 CHECKOUT=50 RELEASE=80. Returns staggered ~2s. Serial leftover.
- Pin held (cannot-evict-pinned=40). 3-live hybrid RS still serializes unique.
- No MET. Next: `LLAMA_HYBRID_RS_CELLS_PAGED` 4 → 24, then 20-way mid-work.

### 2026-08-16 — 16q6 leftover-starve 16/16, QUEUE dark

- Binary `1be7e557b` PID 24475. `.scratch-8k-qwen-overlap-512-16q6/`
- All 16 HTTP 200 cache_n=8000 in 28.4s. Fat tails 137 on 06/08/14/16.
- SWAP=0 QUEUE=0 CHECKOUT=20 RELEASE=32. Leftover restored serially at -np 1.
- Pin still held (cannot-evict-pinned=16). No session-not-found.
- No MET. Next 40q6 on 24475: 40 fat unique must overflow 512+192 leftover.

### 2026-08-16 — 8q6/8q6b pin held, 8 inherit after LRU pressure

- Binary `1be7e557b` PID 24475 `--cache-ram 700` (harness). `.scratch-8k-qwen-overlap-512-8q6/` + `8q6b/`.
- 8q6: grow 8176. Unnamed 8k park HTTP 000. 8× cannot-evict-pinned-named. removing-oldest=0 that cut.
- 8q6b: same PID, no relaunch. All 8 /fork HTTP 200 cache_n=8000 in 14.3s. tokens_cached 8137/8145. No session-not-found.
- Later log: removing oldest 0.001 MiB crumbs; pin skipped 151/666/667 MiB. SWAP=0 QUEUE=0 this-run.
- Pin proved. QUEUE still unproved. No MET. Next 16q6 leftover-starve on 24475.


### 2026-08-16 — 8q5 LRU still dark

- HEAD `1be7e557b`. Same PID 4175. Grow 8176. No "removing oldest". Forks not posted.
- Default prompt cache is no-limit. Two entries cannot prove the pin.
- No MET. Next 8q6: `--cache-ram 700` harness so LRU must fire.


### 2026-08-16 — 8q4 inherit after grow, LRU unproved

- HEAD `1be7e557b`. `.scratch-8k-qwen-overlap-512-8q4/` PID 4175 (fresh).
- Grow 8176. All 8 HTTP 200 cache_n=8000. No session-not-found.
- "removing oldest entry" did not fire. Pin vs prompt-cache LRU unproved.
- No MET. Next 8q5 on same PID: LRU must fire, named session stays.


### 2026-08-16 — 8q3 session lost after tight grow

- HEAD `f4217aca5`. `.scratch-8k-qwen-overlap-512-8q3/` on 67311.
- Master 200 cached=8001. Grow 200 cached=8176 predicted=176.
- All 8 /fork HTTP 400 session not found.
- Log: prompt-cache LRU removed oldest 667 MiB, RELEASE n=510. Named master gone.
- Not QUEUE. No MET. Next 8q4: named session survives grow.


### 2026-08-16 — 40q3 fat unique 40/40

- HEAD `f4217aca5`. `.scratch-8k-qwen-overlap-512-40q3/` on 67311. wall 75s.
- All 40 HTTP 200 cache_n=8000. Four 137-token tails. QUEUE=0. CHECKOUT n=9×40 n=1×4.
- 20–40 HTTP band done (small and fat unique). Queue unproved. No MET. Next 8q3 leftover-starve.


### 2026-08-16 — 16q fat unique 16/16

- HEAD `f4217aca5`. `.scratch-8k-qwen-overlap-512-16q/` on 67311. wall 29s.
- All 16 HTTP 200 cache_n=8000. Four 137-token tails. QUEUE=0. CHECKOUT n=1 for growth.
- No MET. Next 40-way fat unique 90s.


### 2026-08-16 — 8q hang fixed, 8q2 8/8

- HEAD `f4217aca5`. Final prefill n_decoded=1; growth allocate(1) as token delta. Not HTTP starve.
- `.scratch-8k-qwen-overlap-512-8q2/` all 8 HTTP 200 cache_n=8000. child06/08 prompt_n=137.
- CHECKOUT 8×n=9 plus 2×n=1. QUEUE=0. 67311 left up. No MET. Next 16-way fat unique 45s.


### 2026-08-16 — 8q hang is the hole

- HEAD `1d337875`. `.scratch-8k-qwen-overlap-512-8q/` on 86769. 8-block unique, timeout 30s.
- All 8 HTTP 0. One CHECKOUT n=9, populate rid=1 8137/8137, then silence. QUEUE=0.
- 4q 4/4 same unique did not hang. Threshold is 8 live. No MET. Next: hang must become wait-then-finish.


### 2026-08-16 — 4q 8-block no hang

- HEAD `1d337875`. `.scratch-8k-qwen-overlap-512-4q/` on 86769. unique prompt_n=129.
- All 4 HTTP 200 cache_n=8000 in ≤7.4s. QUEUE=0. Hang at 40 not reproduced at 4.
- No MET. Next 8-way 8-block 30s.


### 2026-08-16 — 40q2 hang not QUEUE

- HEAD `1d337875`. `.scratch-8k-qwen-overlap-512-40q2/` on 86769. unique 97–127 tokens (7–8 blocks).
- child05 200 cache_n=8000 prompt_n=113. Other 39 HTTP 0 timed out 300s.
- DS4P-QUEUE=0. CHECKOUT n=8 twice, then 5 min silence. Hang, not overflow-queue.
- Abort storm after timeout; state_read_meta invalid. No MET. Next 4-way 8-block 30s.


### 2026-08-16 — 40q 40/40 no QUEUE

- HEAD `1d337875`. `.scratch-8k-qwen-overlap-512-40q/` on 86769. unique 49–55 tokens (4 blocks).
- All 40 HTTP 200 cache_n=8000. DS4P-QUEUE=0. Leftover still admits every 4-block tail.
- Not the overflow hole. Next 40q2: 6–8 block uniques. No MET.


### 2026-08-16 — Qwen 8k 40-way HTTP inherit

- HEAD `1d337875`. `.scratch-8k-qwen-overlap-512-40/` on 86769. Same 512, -np 1. log 69622 B.
- Master 200 cached=8001. Grow 200 cached=8128.
- All 40 /fork HTTP 200 cache_n=8000. any_500 false. max live HTTP=40.
- DS4P-SWAP 7. DS4P-QUEUE=0. fail_mixed=0. 86769 left up.
- 20–40 HTTP band done. Queue path unexercised. GPU serial. No MET. Next 40q overflow-QUEUE.


### 2026-08-16 — Qwen 8k 20-way HTTP inherit

- HEAD `1d337875`. `.scratch-8k-qwen-overlap-512-20/` on live 78107. Same 512, -np 1.
- Master 200 cached=8001. Grow 200 cached=8128.
- All 20 /fork HTTP 200 cache_n=8000. any_500 false. max live HTTP=20.
- DS4P-SWAP before admit. DS4P-QUEUE=0. fail_mixed=0. 78107 left up.
- Queue path unexercised. GPU serial. No MET. Next 40-way.


### 2026-08-16 — Qwen 8k 16-way HTTP inherit

- HEAD `1d337875`. `.scratch-8k-qwen-overlap-512-16c/` log 60964 B. 512 GPU, watermark 0, -np 1.
- Master 200 cached=8001. Grow 200 cached=8128.
- All 16 /fork HTTP 200 cache_n=8000 n_predicted=8. any_500 false. max live HTTP=16.
- DS4P-SWAP 7 blocks then CHECKOUT. DS4P-QUEUE=0 (leftover after swap enough). fail_mixed=0.
- Queue path unexercised. GPU serial. No MET.
- 16d grow 000 after SIGINT teardown of 71981. Next 20-way on live 78107.


### 2026-08-16 — 16c master died, overflow not run

- HEAD `1d337875` (prepend requeue on `b6dd5aa`). One writer. Not a second process.
- `.scratch-8k-qwen-overlap-512-16c/` master HTTP 000 at 50.8s, empty reply. DS4P-QUEUE/SWAP/fail_mixed none.
- Log: second interrupt + Metal rsets assert on teardown. Then relaunch clobbered the 51k log.
- 18140 later healthy PID 71981. Next is 16d on that serve, new dir. No MET.


### 2026-08-16 — Qwen 8k 16b 15/16

- HEAD `dd2925b6e`. `.scratch-8k-qwen-overlap-512-16b/` log 61083 bytes. All n_predict=8. 512 GPU, watermark 0, -np 1.
- Master 200 cached=8001. Grow 200 cached=8128.
- 16 POSTs before any return. child05 500 outgrew. fail_mixed_remap_once request 1. DS4P-SWAP 7 blocks after the 500.
- Other 15: 200 cache_n=8000. Gate all-16 FAIL.
- No MET. 18140 down.


### 2026-08-16 — Qwen 8k 16-way 14/16

- HEAD `dd2925b6e`. `.scratch-8k-qwen-overlap-512-16/` log 59265 bytes. 512 GPU, watermark 0, -np 1.
- Master 200 cached=8001. Grow 200 cached=8128.
- 16 /fork POSTs in 604us. All posted before any sibling returned. max live HTTP=16.
- child01+02 500 outgrew. fail_mixed_remap_once request 0 and 1. DS4P-SWAP 7 blocks n_past=8112.
- child03–16 200 cache_n=8000. any_500 true. sequential_fail false.
- No /close_session. No MET. 18140 down.


### 2026-08-16 — Qwen 8k 8-way HTTP overlap

- HEAD `dd2925b6e`. `.scratch-8k-qwen-overlap-512-8/` log 79478 bytes. 512 GPU, watermark 0, -np 1.
- Master 200 cached=8001. Grow 200 cached=8128.
- Eight /fork POSTs in 4.2ms at 17:19:34.304–.308 GST. All HTTP 200, all cache_n=8000. any_500 false.
- Eight HTTP live until child2 return 17:19:38.540. child2 first by 18us (sib=false); 1,3–8 sib=true.
- GPU serial: populate 8064, DS4P-SWAP 7 blocks n_past=8112, then 8025, 8072, 8029, 8033, 8033, 8028, 8032.
- No /close_session. No MET. 18140 down.


### 2026-08-16 — Qwen 8k 4-way HTTP overlap

- HEAD `dd2925b6e`. `.scratch-8k-qwen-overlap-512-4/` log 78242 bytes. 512 GPU, watermark 0, -np 1.
- Master 200 cached=8001. Grow 200 cached=8128.
- Four /fork POSTs in 1.3ms at 17:13:40.780–.781 GST. All HTTP 200, all cache_n=8000.
- child1 pred=64 17:13:40.780–17:13:46.225. child2 pred=48 to 17:13:52.224. child3 pred=8 to 17:13:47.177. child4 pred=8 to 17:13:48.134.
- child2/3/4 posted while a sibling was live. any_500 false.
- GPU serial: populate 8072, DS4P-SWAP 7 blocks, then 8028, 8032, 8064.
- No /close_session. No MET. 18140 down.


### 2026-08-16 — Qwen 8k HTTP overlap two /fork

- HEAD `dd2925b6e`. `.scratch-8k-qwen-overlap-512/` log 70017 bytes. 512 GPU, watermark 0, -np 1.
- Master 200 cached=8001. Grow 200 cached=8128.
- child1 200 cache_n=8000 prompt_n=72 pred=64 wall 6.63s (17:07:09.790–17:07:16.418 GST)
- child2 200 cache_n=8000 prompt_n=64 pred=8 wall 1.17s (17:07:09.790–17:07:10.958 GST)
- Both posted the same ms. child2 ended while child1 HTTP still live. Not sequential.
- GPU serial: populate rid=0 8064 (child2), DS4P-SWAP 7 blocks, populate rid=1 8072 (child1).
- No /close_session. No MET. 18140 down.


### 2026-08-16 — Qwen 512 unique-suffix swap child 200

- Landed `dd2925b6e` swap parked unique suffix before fail_mixed remap. `test-paged-kv` ALL PASSED.
- `.scratch-8k-qwen-swap-suffix-512-swap/` log 55520 bytes. `--n-gpu-blocks 512 watermark 0 -np 1 -lv 2`.
- Master 200 `tokens_cached=8001`. Grow 200 `tokens_cached=8128` pred=128. leftover after last allocate=3.
- Child 200 `cache_n=8000` `prompt_n=144` `tokens_cached=8152` pred=8. No /close_session.
- `DS4P-SWAP unique-suffix GPU->CPU: 7 blocks (session not shortened, n_past=8112 table=507)`. leftover 3 < unique 10 ≤ leftover+swapped 10.
- No fail_mixed. No MET. 8k toy-length on a daily model, not 256k.
- child2 200 `cache_n=8000` `prompt_n=128` on live 18140 same master. Checkout n=9 free_before=12. No second SWAP. 18140 down.


### 2026-08-16 — Qwen 512 child 500, no swap

- HEAD `a486fd46f`. `.scratch-8k-qwen-swap-suffix-512/` log 49592 bytes. `--n-gpu-blocks 512 watermark 0 -np 1 -lv 2`.
- Master 200 `tokens_cached=8001`. Checkout n=501 free_before=512. leftover 11.
- Grow 200 `tokens_cached=8048` pred=48. leftover after last allocate=8. No assert.
- Child 500 unique 176 tokens / 12 blocks. `fail_mixed_remap_once` request 0. outgrew. No DS4P-SWAP. No DS4P-EVICT.
- Did not creep to 768. No MET.


### 2026-08-16 — Qwen grow 200; swap slack

- Landed `a486fd46f` named-session grow must not abort as zero-remainder prefill. `test-paged-kv` passed.
- `.scratch-8k-qwen-swap-suffix-768-grow/` log 96197 bytes. Same 768 flags, watermark 0, -lv 2.
- Master 200 `tokens_cached=8001`. Grow 200 `tokens_cached=8048` `tokens_predicted=48` dt 3.64s. No GGML_ASSERT.
- Child 200 `cache_n=8000` `prompt_n=176`. Checkout n=12 free_before=265. No DS4P-SWAP.
- First child 400: 12288 > 8192. Unique large enough to force swap cannot fit in -c 8192 after an 8k prefix.
- 18140 down. No MET.


### 2026-08-16 — Qwen 8k grow assert STOPPED

- HEAD `cd0b32cda`. `.scratch-8k-qwen-swap-suffix-768/` log 96462 bytes. `--n-gpu-blocks 768 --n-cpu-blocks 192 watermark 0 -np 1 -lv 2`.
- Gate 1: master 200, `prompt_n=8000` `tokens_cached=8001` `tokens_predicted=1`. Checkout n=501 free_before=768. Then RELEASE n=501.
- Gate 2 grow: second `/completion` same session `n_predict=48` empty reply 0.10s. `GGML_ASSERT remaining_prompt > 0` at llama-paged-scheduler-impl.cpp:1319.
- No DS4P-SWAP. No child. 18140 down. No MET.


### 2026-08-16 — Qwen 8k unique-suffix STOPPED

- HEAD `cd0b32cda`. `.scratch-8k-qwen-swap-suffix/` log 97110 bytes. `-np 1 -c 8192 --n-gpu-blocks 504 --kv-paged-watermark 0 -lv 2`.
- Master 500 `outgrew the block pool` (dt 52.3s). Checkout n=501 free_before=504. `fail_mixed_remap_once` on request 0. RELEASE n=501.
- Child 200 `cache_n=8000` `prompt_n=39` after master died. Checkout n=3 free_before=4. No DS4P-SWAP. Not this proof.
- Probe: connection closed. GGML_ASSERT remaining_prompt > 0 at llama-paged-scheduler-impl.cpp:1319.
- 18140 down. 8048 untouched. No MET.


### 2026-08-16 — tight-GPU mixed-table STOPPED

- HEAD `cd0b32cda`. Implementer STOP: child 200 at wm0/wm2 needs a dual-buffer Metal attention kernel (parked).
- `get_kv_tensor` returns only `kv_gpu_layers[i]` (llama-kv-cache-paged.cpp:903). CPU layers exist for `do_block_copy`, not decode.
- `evict()` skips master, named sessions, mixed tables (`count_cpu_unique > 0`), unref==0, then `no unref-tail victim (master prefix stays.)`.
- On prefix-full there is no non-prefix victim. Child unique already on CPU frees no GPU.
- wm8 child 200 was watermark slack. Slack child2 is not a proof. No MET.


### 2026-08-16 — wm8 prefix-full child 200

- Same HEAD `cd0b32cda`. `--n-gpu-blocks 510 --kv-paged-watermark 0.015686` (8 blocks), `-np 1`, stories15M 8k, port 18140, `-lv 2`.
- Log `.scratch-8k-prefix-full-wm8/server.log` 11985 bytes. Two launches in that file; proof is the second (clock reset).
- Master 200, `tokens_cached=8000`, n_past 8000. Child 200, `cache_n=7984` `prompt_n=28` `tokens_predicted=8`. No /close_session.
- Stamps: allocate n=2 free_before=11, mixed-scratch 9/8, RELEASE n=2 (repeats), RELEASE n=501. No DS4P-MIXED, no fail-once.
- Prefix GPU ids not logged at -lv 2. Old 6.9GB and wm2 logs untouched.
- No reserve_decode_scratch. No MET. Toy 8k, not a daily model.


### 2026-08-16 — cut cd0b32cda: fail-once; wm2 child 500

- Landed `cd0b32cda` fail mixed-remap child once, do not spin.
- `test-paged-kv` ALL PASSED including `test_mixed_remap_fail_once_does_not_spin`.
- wm2 proof: `--n-gpu-blocks 502 --kv-paged-watermark 0.003984`, port 18140, `-lv 2`.
- Master 200, `tokens_cached=8000`. Child 500 `outgrew the block pool`.
- Log `.scratch-8k-prefix-full-wm2/server.log` 5946 bytes. `fail_mixed_remap_once` once. 6.9GB file untouched.
- Remap checked out mixed-scratch 3→2 then needed more GPU without touching the prefix.
- No reserve_decode_scratch. No MET. Child never 200.


### 2026-08-16 — prefix-full mixed-table STOPPED

- HEAD `e4f341aa9`. Paged files clean. No reserve_decode_scratch.
- stories15M 8k, `-np 1`, n_gpu_blocks=500, watermark=0, port 18140.
- Log stamps (`.scratch-8k-prefix-full/stamp-first.txt`): fork 499 blocks / 7984 tokens; `DS4P-MIXED unique suffix on CPU: 2 block(s)`; `remap needs more GPU than can be freed without touching the master prefix`.
- Child never 200 (timeout / later retry 400 session-not-found). Do not stamp from `summary.json` (retry trash).
- `server.log` grew to 6.9GB spinning that ERROR at `-lv 4`. Serve stopped. Log kept.
- Does NOT stamp MET. Bar still NOT MET.


### 2026-08-16 — cut d5ffab3aa: unique-suffix swap

- Landed `d5ffab3aa` paged: swap a named hold's unique suffix to CPU.
- stories15M 8k, `-np 1`, port 18140, n_gpu_blocks=504, n_cpu_blocks=64.
- Log: `DS4P-SWAP unique-suffix GPU->CPU: 3 blocks (session not shortened, n_past=8048 table=503)`.
- Child HTTP 200 without `/close_session`. `cache_n=8000` `prompt_n=40` `tokens_evaluated=8040`.
- Probe after child still `cache_n=8000`.
- Shared prefix stayed GPU (500 blocks / 8000 tokens). Master GPU IDs for those blocks not rewritten.
- JSON in `llama.cpp-ds4ports/.scratch-8k-swap-suffix/`. server.log 56494 bytes.
- HONEST: unique suffix only, stories15M not a daily model. Mixed-table child still blocked (`get_kv_tensor` GPU-only).
- Does NOT stamp MET. Bar still NOT MET.


### 2026-08-16 — dfd4800f6 re-proof: 4-cell RS in live log

- Re-proof on HEAD `dfd4800f6`, port 18774, `-np 1` `-c 8192`.
- Live log `.scratch-8k-qwen-reproof/server.log` 89785 bytes (append-only).
- Log: `llama_memory_recurrent: size = 598.50 MiB (4 cells, 64 layers, 1 seqs 0 rs_seq)`.
- Log: froze hybrid RS prefix on hold seq 3 from src 0 (`qwen-8k-reproof-master`, rs_size=4).
- Log: growing slot id=1 into few live RS cells (rs_size=4 hold=3).
- Log: froze hold seq 3 from src 1 (`qwen-8k-reproof-seq-master`).
- No defer / not-growing lines.
- Overlap JSON: master 200 cache_n=0 prompt_n=133 id_slot=0; child1 200 cache_n=128 prompt_n=12 id_slot=0; child2 200 cache_n=128 prompt_n=13 id_slot=1; unknown 400. Wall clocks overlap.
- Sequential own JSON: both children cache_n=128, same id_slot=1.
- Still a 3-live+hold slot wall, not pool admission. 8k is not 256k.
- `-np` unchanged. Fleet not touched. MET_stamped=false.
- Does NOT stamp MET. Bar still NOT MET.


### 2026-08-16 — cut dfd4800f6: HTTP inherit only (RS size unproven)

- Reviewer: did not move the bar. Stamped from JSON, not writeup.
- master 200 `cache_n=0` `prompt_n=133` `id_slot=0`
- child1 200 `cache_n=128` `prompt_n=12` `id_slot=1`
- child2 200 `cache_n=128` `prompt_n=13` `id_slot=0`
- unknown 400
- `-np` 1. Not pushed.
- `id_slot=1` is the only grow proof. `server.log` 73 bytes (wiped). `summary.json` rewritten, not prove.py output.
- Will not stamp 4 cells / 598.5 MiB / hold seq 3 / no-defer / sequential reuse from a missing log.
- Still a slot wall, not pool admission. 8k is not 256k. No MET.
- JSON in `llama.cpp-ds4ports/.scratch-8k-qwen-few-rs/`.


### 2026-08-16 — DSV4 Flash 0731 8k two-child named /fork

- DSV4 Flash 0731 8k two-child named /fork.
- Both `cache_n=128`.
- Overlap deferred at `n_seq_max=1`, no crash.
- Master still forkable.
- Unknown 400.
- JSON in `llama.cpp-ds4ports/.scratch-dsv4-8k-two-child/`.
- HONEST: queue-then-serial, not concurrent usable serve.
- Sink layers still static.
- 8k is not 256k.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — hybrid RS widen STOPPED

- Hybrid RS widen STOPPED.
- Not DSV4 bookkeeping-only.
- Qwen RS cell is 150 MiB; 256 cells ~37 GiB.
- Sharing one cell breaks SSM rewind.
- HEAD `06fba44e2`.
- `2474fd5fb` defer remains the Qwen two-child path at `-np 1`.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — cut 06fba44e2: --kv-paged default-on for Qwen35/QWEN35MOE only

- Landed `06fba44e2` `--kv-paged` default-on for Qwen35/QWEN35MOE only.
- 8k /fork env unset: `cache_n=128` `prompt_n=12` `tokens_evaluated=140`.
- Other hybrids still opt-in (silent-static risk).
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — two children named /fork from one 16k Qwen3.8 master

- Two children named /fork from one 16k Qwen3.8 master.
- Both `cache_n=16384`.
- Overlap crash fixed `2474fd5fb` (defer second hybrid child past `n_seq_max`).
- Master still forkable after.
- Unknown 400.
- JSON in `llama.cpp-ds4ports/.scratch-two-child-16k/`.
- HONEST: serial/defer at `-np 1`, not a concurrent usable serve. Not 256k.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — Qwen3.8 27B -c 32768 28k-prefix named /fork

- Qwen3.8 27B `-c 32768` 28k-prefix named /fork.
- Master 28672 tokens / 306.4s.
- Child `cache_n=28672` `prompt_n=9` `tokens_evaluated=28681` in 1.66s.
- 1792 blocks by reference.
- Unknown 400.
- JSON in `llama.cpp-ds4ports/.scratch-28k-named-fork/`.
- HONEST: 28k is not a 32k fill, not 256k.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — Qwen3.8 27B -c 32768 24k-prefix named /fork

- Qwen3.8 27B `-c 32768` 24k-prefix named /fork.
- Master 24576 tokens / 241.6s.
- Child `cache_n=24576` `prompt_n=9` `tokens_evaluated=24585` in 1.25s.
- 1536 blocks by reference.
- Unknown 400.
- JSON in `llama.cpp-ds4ports/.scratch-24k-named-fork/`.
- HONEST: 24k is not a 32k fill, not 256k.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — Qwen3.8 27B -c 32768 16k-prefix named /fork

- Qwen3.8 27B `-c 32768` 16k-prefix named /fork.
- Master 16384 tokens / 135.2s.
- Child `cache_n=16384` `prompt_n=9` `tokens_evaluated=16393` in 1.04s.
- 1024 blocks by reference.
- Unknown 400.
- JSON in `llama.cpp-ds4ports/.scratch-16k-named-fork/`.
- HONEST: 16k is not a 32k fill, not 256k.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — Qwen3.8 27B -c 32768 long-prefix named /fork

- Qwen3.8 27B `-c 32768` long-prefix named /fork.
- Master 9241 tokens / 63.8s.
- Child `cache_n=9232` `prompt_n=21` `tokens_evaluated=9253` in 0.92s.
- 577 blocks by reference.
- Unknown 400.
- JSON in `ornith-1m/_scratch/qwen38-fork-32k-long-18091/`.
- HONEST: 9k is not a 32k fill, not 256k.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — Qwen3.8 27B -c 32768 named /fork HTTP started

- HEAD `a7359a75f`.
- Qwen3.8 27B `-c 32768` named /fork HTTP started.
- `n_gpu_blocks=3072`, no overcommit.
- Child `cache_n=32` `prompt_n=6` `tokens_evaluated=38`.
- Unknown 400.
- RSS ~22 GiB.
- JSON in `/tmp/qwen38-fork-32k-18091/`.
- HONEST: 32-token prompt is not a 32k-context proof.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — Qwen3.8 27B 8k named /fork HTTP proven

- HEAD `a7359a75f`.
- Qwen3.8 27B 8k named /fork HTTP proven.
- Child `cache_n=128` `prompt_n=12` `tokens_evaluated=140`.
- Child2 `cache_n=128` `prompt_n=16` `tokens_evaluated=144`.
- Unknown 400.
- `DS4P_PAGED_HYBRID=1`, `n_gpu_blocks=768`, all attention layers paged path.
- JSON in `llama.cpp-ds4ports/.fork-proof-scratch/`.
- Does NOT stamp MET. 8k is not 256k. Bar still NOT MET.

### 2026-08-16 — DSV4 Flash 0731 8k named /fork HTTP proven

- HEAD `a7359a75f`, new binary.
- DSV4 Flash 0731 8k named /fork HTTP proven.
- Child `cache_n=64` `prompt_n=22` `tokens_evaluated=86`.
- Unknown 400.
- `n_gpu_blocks=192` no overcommit.
- Does NOT stamp MET. 8k is not 256k. Bar still NOT MET.

### 2026-08-16 — NEW pool-full HTTP proven on stories15M

- HEAD `a7359a75f`, no new code.
- Children waited; named hold not shortened (`after_fork` `cache_n=112`).
- `/close_session` then children 200.
- evicted unique-suffix from named hold: 0.
- Old 8+4 200s (master tail evicted) are dead.
- Does NOT stamp MET. 256-token ctx is not 256k. Bar still NOT MET.

### 2026-08-16 — cut a7359a75f: named master filling GPU, children WAIT not CPU swap

- Landed `a7359a75f` on llama.cpp-ds4ports `ds4-ports`.
- Named master filling GPU: children WAIT (not CPU swap, not 500) until `/close_session`.
- `allocate()` is GPU-only.
- Shared-prefix child cannot whole-table swap.
- `test_named_master_full_gpu_children_wait_not_cpu_swap` passed.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — cut 257458bdd: named/held master prefix is not an eviction victim

- Landed `257458bdd` on llama.cpp-ds4ports `ds4-ports`.
- Named/held master prefix is not an eviction victim.
- `evict_held_prefix` skips session holds; child queues if that is the only reclaim.
- `test_named_master_not_eviction_victim`: hold stayed 4 blocks, child inherited 64 not 32.
- `test-paged-kv` ALL PASSED.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — HTTP pool-full proven on stories15M

- HEAD `6911a61ac`, no new code.
- `--kv-paged -np 1 -c 256`, 8 gpu + 4 cpu blocks.
- Session stayed; shared prefix stayed; master's unique suffix was evicted (112 → 64).
- 3 children `/fork` waited then 200, `cache_n=64`.
- after_fork inherited 64 of 112 (same session 200). Queue+preempt-tail, not full master survived.
- Logs: waiting=3, evict unique-suffix from held prefix, no 500.
- Does NOT stamp MET. 256-token ctx is not 256k. Bar still NOT MET.

### 2026-08-16 — 256 bookkeeping is seq_id / bitset width, not a leftover -np slot wall

- `DS4P_PAGED_MAX_BOOKKEEPING = LLAMA_MAX_SEQ` is seq_id / bitset width, not a leftover `-np` slot wall.
- Sequential named sessions recycle idle slots.
- 257th concurrent in-flight defers (queue), not refuse/crash/drop.
- Did not lift the cap. Did not add a loud refuse (that would create a wall).
- HEAD still `6911a61ac`. Bar still NOT MET.

### 2026-08-16 — cut 6911a61ac: paged named /fork prompt_n

- Landed `6911a61ac` on llama.cpp-ds4ports `ds4-ports`.
- Paged named /fork does not bill inherited tokens as `prompt_n`.
- stories15M child JSON: `cache_n=32`, `prompt_n=7`, `tokens_evaluated=39`. Parent cold `cache_n=0`.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — cut d11c037f3: named /fork HTTP cache_n

- Landed `d11c037f3` on llama.cpp-ds4ports `ds4-ports`.
- Child named /fork now reports inherited tokens as HTTP `cache_n` (was always 0 on paged).
- stories15M child `cache_n=32`.
- `test-paged-kv` ALL PASSED including `test_named_fork_n_past_is_http_cache_n`.
- Remaining: paged `prompt_n` still counts inherited tokens (`prompt_n + cache_n != n_prompt`). APC/warm `cache_n` still unset.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — cut 9f12eb160: n_gpu_blocks==0 refuse (not abort)

- Landed `9f12eb160` on llama.cpp-ds4ports `ds4-ports`.
- `kv_paged` refuses `n_gpu_blocks==0` instead of GGML_ASSERT abort. `llama_context` ctor throws; server exits 1.
- `test-paged-kv` ALL PASSED including `test_init_zero_gpu_blocks_throws`.
- Does NOT stamp MET. 32k DSV4 still FAIL. Bar still NOT MET.

### 2026-08-16 — 32k DSV4 named /fork FAIL

- HEAD `95e35281c` comments/tests only. Binary `ab2507c5e`.
- `-c 32768`: `common_fit_paged_kv_blocks` requested n_ctx=32768 needs 768 KV blocks (4.0 GiB), budget 231 (1.2 GiB). Largest n_ctx that fits ~9856. Then GGML_ASSERT `n_gpu_blocks==0` in `llama-kv-cache-paged.cpp`.
- 16k vanilla same refuse (384 vs 231).
- 16k with `--n-gpu-blocks 384`: named /fork SHARED worked (master 128 inherit, two serial children, unknown 400). RSS 92.55 GiB. Useful proof, NOT a 32k pass.
- Implication: 256k–1M DSV4 is box work. Mac ceiling after ~90G weights is ~10k KV unless overcommit.

## How we work

Allowed now (Mac in use):
- Read, plan, code, unit tests, tiny-model e2e (stories15M-class).
- Cheap "3 overlapping HTTP, `-np 1`, queue not reject" gates.

Parked until Satinder says the Mac is free:
- NIAH
- 256k / 512k / 1M ABBA ladders (the live 1M **fill + /fork** is the product gate he unparked, not NIAH)
- Decode-curve rungs / Metal champion kernels (after the 1M `/fork` proof)

Parked until the 2×96GB box is this seat's:
- CUDA parity, hd512 champion, hybrid CUDA witness
- 256k–1M DSV4 (Mac ceiling after ~90G weights is ~10k KV unless overcommit)

Never:
- 86-arch / 95-arch / 141-arch sweeps
- Raising `-np` to fake concurrency
- Marking P2-8 closed on a toy gate
- Calling 1k–4k "done"

## Next slice

1. 1M fill aborted. Inherit unverified. `/fork` for real use. No MET.
2. 20/50/100 spray skipped.
3. Then Metal batched decode (multi-seq one graph). Not `-np`. Not CUDA. Not on this fill. Days if existing Metal FA packs; 1–3 weeks if new paged kernels.
4. **Product:** stock `/v1/chat/completions` auto-shares matching prefixes (vLLM/SGLang). `/fork` is optional, not the client contract. Satinder 2026-08-17 18:35 GST.

## Daily models (these outrank the 19)

- Qwen3.8 27B only (1M paged fill + 20/50/100 `/fork` is the live gate)
- DeepSeek V4 Flash 0731 (parked 2026-08-17)
- GLM 5.2 (box, later)

Aug 7 19-list stays parked-to-last except where it overlaps.

## Short list (park everything else)

Aug 7: deepseek v4, dflash, eagle3, ernie4-5, ernie4-5-moe, gemma4-assistant, grok, hunyuan-moe, hunyuan-vl, laguna, mimo2, minimax-m3, nemotron, qwen3moe, qwen3next, qwen3vl, qwen3vlmoe, starcoder, step35.
