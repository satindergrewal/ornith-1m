#!/usr/bin/env python3
"""loop_battery.py — deep-context loop reproduction battery for chat LLMs.

Purpose: detect degenerate repetition loops ("attractor loops") that appear
in long-horizon sessions — deep context plus heavy multi-step reasoning —
the failure class observed on aggressive weight quantization at 600k-class
depths. NOT a needle test: it never asks about recall; it builds realistic
deep sessions and checks the GENERATION for structural loop signatures.

Method
------
1. Build a deep session: a stack of realistic filler documents (mixed
   domains, code, tables, prior task transcripts) up to a target token
   depth, ending with one DEEP-REASONING task from cases/ (the class that
   triggers the failure: compare/derive/prove-two-ways-then-judge, which
   forces long chained reasoning; short factual chats do not trigger it).
2. Generate n samples (temp configurable) plus one greedy pass.
3. Detect loops STRUCTURALLY on the generated tokens:
     a. exact-cycle: the tail is periodic — minimal period p where the last
        8*p tokens equal the last p tokens repeated 8 times.
     b. repeated-span coverage: the most-repeated >=20-token n-gram covers
        >= 60% of the last 1024 generated tokens with >= 8 repeats.
     c. entropy collapse (supporting only): mean next-token entropy of the
        final 200 tokens below threshold.
   Verdict LOOP if (a) or (b). Verbose-but-novel text never trips these.

Usage
-----
  python3 loop_battery.py --base-url http://HOST:8012/v1 --model NAME \
      --api-key KEY [--depths 350000,500000,650000] [--samples 3] \
      [--max-tokens 4096] [--out results/] [--cases cases/]

Any OpenAI-compatible /v1 endpoint works (vLLM, llama.cpp server, gateways).
Results: JSONL rows + a per-depth/per-case loop matrix summary. Publish
cases you add alongside; cases are plain JSON.
"""

import argparse
import json
import math
import os
import time
import urllib.request

# ---------------------------------------------------------------- filler bank
# Realistic multi-domain session filler (deterministic; expanded on demand).
_FILLER_DOCS = [
    ("systems-notes", """Distributed systems review notes, session {i}.
Consensus: Raft separates leader election, log replication, and safety. A
term is a logical clock; each election increments it. Log matching property:
if two logs share an index and term, all preceding entries match. Commit
rule: an entry is committed once replicated to a majority and a later entry
from the same leader replicates on top of it. Failure detectors in practice
are timeouts; suspicion mechanisms (phi accrual) reduce false positives. In
our cluster of {n} nodes the measured p99 election time was 380 ms with a
150 ms base timeout."""),
    ("code-review", """Code review transcript, task {i}.
Reviewed a patch replacing a recursive descent parser with a table-driven
one. Findings: (1) the shift table is rebuilt per token; hoist it. (2) Error
recovery drops two tokens on a failed reduce; the old parser dropped one.
(3) The benchmark harness warms the cache once but reports cold medians.
Verdict: request changes, the error-recovery regression is user visible on
malformed input near EOL."""),
    ("physics-ref", """Reference extract, thermodynamics {i}.
The Helmholtz free energy A = U - TS is the natural potential at fixed T, V;
the Gibbs free energy G = U + PV - TS at fixed T, P. Spontaneity at constant
pressure tracks dG < 0. For a magnetic system in field H the relevant work
term is -M dH, so the potential picks up an MH term. Tables used below list
heat capacities at 298 K for seventeen elements."""),
    ("log-excerpt", """Ops log excerpt {i}.
03:{m}:12 scheduler replay lag 2.1s (threshold 5s) - no action
03:{m}:40 cache hit ratio 0.71 (baseline 0.68)
04:{m}:05 network hiccup, 3 retried RPCs, all succeeded
04:{m}:59 disk fill 71% on shard 2, watermark 80%
Recurring theme this hour: none. Capacity headroom nominal."""),
    ("prior-task", """Prior session task {i}: "summarize the tradeoffs of
event sourcing versus CRUD for an audit-critical ledger." Assistant answer
summary: event sourcing gives a complete append-only history and time-travel
reconstruction, at the cost of schema evolution complexity and replay time;
CRUD is simpler and faster to ship but loses intermediate states. Final
recommendation was hybrid: CRUD projection over an event spine."""),
]


def build_filler(target_tokens, approx_tok_per_char=0.28):
    """Deterministic filler text up to roughly target_tokens tokens."""
    out = []
    i = 0
    est_chars = target_tokens / approx_tok_per_char
    while sum(len(d) for d in out) < est_chars:
        kind, body = _FILLER_DOCS[i % len(_FILLER_DOCS)]
        out.append(body.format(i=i, n=5 + (i % 4), m=(i * 7) % 60))
        i += 1
    return "\n\n".join(out)


# ---------------------------------------------------------------- detection
def exact_cycle(tokens, max_period=4096, reps=8):
    """Minimal period p such that the last reps*p tokens equal the last p
    tokens repeated. Returns (period, span) or (None, 0)."""
    T = len(tokens)
    for p in range(1, min(max_period, T // reps) + 1):
        tail = tokens[T - p:]
        span = p * reps
        if T >= span and tokens[T - span:] == tail * reps:
            # confirm minimal
            return p, span
    return None, 0


def repeated_span_coverage(tokens, ngram=20, window=1024, min_repeats=8):
    """Most-repeated n-gram coverage of the tail window. Returns
    (coverage, ngram_tokens, repeats) of the worst offender."""
    w = tokens[-window:]
    if len(w) < ngram * min_repeats:
        return 0.0, None, 0
    counts = {}
    for i in range(len(w) - ngram):
        key = tuple(w[i:i + ngram])
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return 0.0, None, 0
    key, rep = max(counts.items(), key=lambda kv: kv[1])
    cov = (rep * ngram) / len(w)
    return cov, list(key), rep


def detect_loop(gen_tokens, logprobs=None):
    p, span = exact_cycle(gen_tokens)
    cov, ngram, rep = repeated_span_coverage(gen_tokens)
    loop = (p is not None) or (cov >= 0.60 and rep >= 8)
    ent = None
    if logprobs and len(logprobs) >= 200:
        tail = [x for x in logprobs[-200:] if x is not None]
        if tail:
            ent = -sum(tail) / len(tail)
    return {"loop": bool(loop), "cycle_period": p, "cycle_span": span,
            "repeat_coverage": round(cov, 3), "repeat_count": rep,
            "tail_entropy": round(ent, 3) if ent is not None else None}


# ---------------------------------------------------------------- api client
class Client:
    def __init__(self, base, key, model):
        self.base, self.key, self.model = base, key, model

    def chat(self, messages, max_tokens, temperature, seed=None):
        body = {"model": self.model, "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature,
                "logprobs": True, "top_logprobs": 1}
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(
            self.base.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        t0 = time.time()
        d = json.load(urllib.request.urlopen(req, timeout=3600))
        msg = d["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        lps = [c["logprobs"]["content"][0]["logprob"]
               if c.get("logprobs") and c["logprobs"].get("content")
               else None for c in d["choices"][0].get("logprobs", {}).get("content", [])]
        return {"content": content, "reasoning": reasoning,
                "logprobs": lps, "wall": round(time.time() - t0, 1),
                "usage": d.get("usage", {})}

    def tokenize(self, text):
        req = urllib.request.Request(
            self.base.rstrip("/") + "/tokenize",
            data=json.dumps({"model": self.model, "prompt": text}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        try:
            return len(json.load(urllib.request.urlopen(req, timeout=60))["tokens"])
        except Exception:
            # char-based fallback estimate
            return int(len(text) * 0.28)


# ---------------------------------------------------------------- battery
def run_case(client, case, depth, args):
    filler = build_filler(depth - client.tokenize(case["task"]))
    messages = [{"role": "user",
                 "content": filler + "\n\n=== NEW TASK (answer now) ===\n\n"
                 + case["task"]}]
    rows = []
    gens = [(f"s{s}", args.temperature, 1000 + s) for s in range(args.samples)]
    gens.append(("greedy", 0.0, None))
    for tag, temp, seed in gens:
        try:
            r = client.chat(messages, args.max_tokens, temp, seed)
        except Exception as e:
            rows.append({"case": case["name"], "depth": depth, "gen": tag,
                         "error": str(e)[:200]})
            continue
        toks = client.tokenize(r["content"] + r["reasoning"])
        # tokenize() counts the prompt class; for detection we re-derive
        # tokens from logprob entries when present, else approximate by
        # whitespace words (cycle/coverage on words is equally valid).
        words = (r["reasoning"] + " " + r["content"]).split()
        det = detect_loop(words, r["logprobs"])
        rows.append({"case": case["name"], "depth": depth, "gen": tag,
                     "detect": det, "n_words": len(words),
                     "wall": r["wall"],
                     "usage": r["usage"],
                     "tail": " ".join(words[-30:])})
        print(f"[{case['name']} @{depth} {tag}] loop={det['loop']} "
              f"cov={det['repeat_coverage']} cyc={det['cycle_period']} "
              f"words={len(words)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--cases", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cases"))
    ap.add_argument("--depths", default="350000,500000,650000")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    client = Client(args.base_url, args.api_key, args.model)
    cases = [json.loads(l) for l in open(os.path.join(args.cases, "cases.jsonl"))]
    depths = [int(d) for d in args.depths.split(",")]
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.out, f"battery-{stamp}.jsonl")

    n_loop = n_total = 0
    with open(path, "w") as f:
        for case in cases:
            for depth in depths:
                for row in run_case(client, case, depth, args):
                    f.write(json.dumps(row) + "\n")
                    n_total += 1
                    if row.get("detect", {}).get("loop"):
                        n_loop += 1
    print(f"\nBATTERY DONE: {n_loop}/{n_total} generations LOOPED")
    print(f"results -> {path}")


if __name__ == "__main__":
    main()
