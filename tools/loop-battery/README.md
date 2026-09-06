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

License: same as the repository.
