# Loop Reproduction Battery

Detects degenerate repetition loops ("attractor loops") in long-horizon LLM
sessions: the failure class where a model at very deep context (500k-class)
under heavy multi-step reasoning falls into a repeating token cycle and
never terminates. Observed in the wild on aggressively quantized open
weights; short factual chats do not trigger it.

This is NOT a needle-in-a-haystack recall test. It builds realistic deep
sessions (mixed documents, code reviews, prior task transcripts) ending in
deep-reasoning tasks from `cases/cases.jsonl`, then checks the GENERATION
for structural loop signatures:

1. **exact cycle** - the generated tail is periodic (minimal period p with
   the last 8p tokens equal to the last p tokens repeated 8 times)
2. **repeated-span coverage** - a >= 20-token n-gram covering >= 60% of the
   last 1024 tokens with >= 8 repeats
3. **entropy collapse** (supporting signal only) - mean next-token logprob
   entropy of the final 200 tokens

A verbose but novel answer never trips (1) or (2). Verdict per generation:
LOOP or CLEAN, with the repeating span as the receipt.

## Usage

Any OpenAI-compatible endpoint (vLLM, llama.cpp server, gateways):

```
python3 loop_battery.py --base-url http://your-host:8012/v1 \
    --model YOUR-MODEL-NAME --api-key YOURKEY \
    --depths 350000,500000,650000 --samples 3 --max-tokens 4096
```

Cost note: each (case, depth) cell prefills a deep context and generates up
to `--max-tokens`; a full battery is an hours-class run against a local
serve. Start with one depth to profile.

## Cases

`cases/cases.jsonl` - six deep-reasoning prompts across code comparison
(quicksort in C vs assembly - the original live trigger class), science
derivations, proofs, engineering design, systems, and one
abliteration-boundary prompt (a stock model hedges or refuses it; an
abliterated model engages - deep reasoning under the refusal-direction
load). Add your own; cases are plain JSONL.

`cases-cybergym/cases.jsonl` - eight security-research reasoning prompts:
p2p protocol audit (QUIC mux over iroh/quinn), public CVE root-cause
chains with detection-first PoCs (Leaky Vessels class), container
namespace/seccomp boundary tracing, privilege-escalation chain analysis
from an unprivileged shell, hybrid post-quantum handshake design
(X25519 + ML-KEM, Noise IK binding), fuzzer crash triage to root cause,
dependency-resolution supply-chain attacks, and speculative-execution
side-channel reasoning with detection microbenchmarks. This is the
daily-workload class for protocol exploit research on p2p codebases:
sustained multi-hundred-step causal reasoning, which is exactly the
generation shape that triggers the deep-context attractor. PoC-writing
prompts are detection-first (auditor tooling for isolated labs), not
weaponization.

Run a specific set with `--cases cases-cybergym`. For a before/after
(base vs capped) comparison, run the SAME case set and depths against
both endpoints and diff the per-case loop rates.

## Tool-use battery mode (agentic sessions)

Plain generation is one failure surface. Deep agentic sessions (the real
workload) add another: repeated identical tool calls, circular search
loops, and re-querying the same thing forever are the same attractor
wearing a tool harness. `tool_battery_agentic.py` gives the model a
`web_search` tool (served by the harness over the Brave Search API - the
model never sees the key; it is read from `--brave-key-file` or
`$BRAVE_API_KEY` at runtime and must never be committed) and runs the
same deep-context research cases. Detection runs on BOTH channels:

1. **tool-call signatures** - exact cycle, single-call dominance, and
   repeated n-gram coverage over the (tool, args) sequence
2. **assistant text** - the same detectors as the text battery, run over
   reasoning + content

Verdict per session: `TOOL_LOOP`, `TEXT_LOOP`, or `CLEAN`, with the
repeating call/query span as the receipt.

```
python3 tool_battery_agentic.py --base-url http://your-host:8012/v1 \
    --model YOUR-MODEL --api-key KEY \
    --brave-key-file /path/outside/this/repo/brave.key \
    --depths 350000,500000 --cases cases-cybergym --max-steps 24
```

Keep the Brave key file OUTSIDE this repository (the key is read at
runtime and must never be committed; a rejected 401/403 key aborts the
run immediately instead of burning sessions). `--selftest` validates the
detectors offline. Each output row carries `verdict` (TOOL_LOOP /
TEXT_LOOP / CLEAN / ERROR), `truncated` (finish_reason=length - a
budget-limited session is not a completed answer), the tool and text
detector details, and the query log; ERROR rows are harness/server
failures, excluded from the headline loop count.

Brave's free tier is 1 query/sec: the harness throttles every live
attempt (failures included), retries a 429 once with backoff, and caches
successful queries for the whole run (deterministic filler means repeat
sessions re-ask the same queries). Case prompts transfer unchanged from
the text battery sets (the cybergym set is the natural fit - CVE
research, protocol audits).

## Interpreting results

- `loop: true` on any generation at a depth = the artifact loops at that
  depth. Save the `tail` field (last 30 words) as the reproduction receipt.
- Loops that appear only in `reasoning` (thinking) content but not the
  final answer still count - pass them through; detection runs on the
  concatenation.
- Day-to-day flakiness is expected: the attractor is context-state
  dependent. `--samples 3` plus greedy gives four shots per cell.

## Method notes

- Filler is deterministic (same battery = same contexts, reproducible).
- Detection runs on whitespace words when server logprobs are unavailable;
  cycle and coverage detection on words is equivalent for this failure
  class (the loops observed repeat multi-word spans).
- The battery is endpoint-agnostic on purpose: run the same battery against
  a local quant, an API model, or a before/after pair (base vs capped) to
  measure whether an intervention (finetune, requant, ThinkingCap-style
  termination training) reduces looping.
