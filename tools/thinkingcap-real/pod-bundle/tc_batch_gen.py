#!/usr/bin/env python3
"""tc_batch_gen.py — ThinkingCap REAL STEP 2: on-policy regen (batched).

Coordinator order 2026-09-04: regen kind-2 (a+b*c) only at 160-token budget,
same temp/samples/template; re-grade; merged downstream with kind/budget
provenance. Rows carry: kind, budget, hit_cap, correct (strict last-int) and
correct_first_answer (first-line answer regrade, for truncated-after-answer
cases only — strict stays the graded field).

Full-run history + seeds in PROGRESS.md.
"""

import argparse
import json
import os
import random
import re
import time

import torch

from full_loader import StreamingDSV4
from dsv4_full import DSV4Full

TRAIN_SEED = 20260904
HOLDOUT_SEED = 20260905
GLOBAL_SEED = 20260977
TEMP = 0.8

# Official DeepSeek-V4 prompt format (checkpoint encoding/encoding_dsv4.py,
# thinking_mode="chat" i.e. non-thinking: direct answer after </think>).
BOS = "<｜begin▁of▁sentence｜>"
USER_SP = "<｜User｜>"
ASSISTANT_SP = "<｜Assistant｜>"
THINK_END = "</think>"


def build_prompt(q):
    return BOS + USER_SP + q + ASSISTANT_SP + THINK_END


def make_problems(seed, n):
    rng = random.Random(seed)
    probs = []
    for i in range(n):
        a, b, c = rng.randint(2, 49), rng.randint(2, 49), rng.randint(2, 9)
        kind = i % 3
        if kind == 0:
            q, ans = f"What is {a} + {b}?", a + b
        elif kind == 1:
            q, ans = f"What is {a} * {c}?", a * c
        else:
            q, ans = f"What is {a} + {b} * {c}?", a + b * c
        probs.append({"pid": i, "kind": kind, "q": q, "ans": ans})
    return probs


def parse_answer(text):
    m = re.findall(r"-?\d+", text)
    return int(m[-1]) if m else None


def first_answer(text):
    m = re.findall(r"(?:=|\bis\b|\*\*)\s*\**\s*(-?\d+)", text.split("\n")[0])
    return int(m[0]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--kinds", type=str, default="0,1,2")
    ap.add_argument("--seed", type=int, default=TRAIN_SEED)
    ap.add_argument("--sample-seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--out", type=str, default="/wd/data/regen.jsonl")
    ap.add_argument("--tag", type=str, default="regen")
    a = ap.parse_args()
    kinds = {int(k) for k in a.kinds.split(",")}

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("/model/tokenizer.json")

    probs = [p for p in make_problems(a.seed, a.n) if p["kind"] in kinds]
    L = StreamingDSV4()
    eng = DSV4Full(L)
    eos = L.cfg.get("eos_token_id", 1)

    seqs = []
    for p in probs:
        base = tok.encode(build_prompt(p["q"])).ids
        for s in range(a.samples):
            seqs.append({"pid": p["pid"], "kind": p["kind"], "s": s,
                         "ids": list(base), "done": False, "new": []})
    print(f"[{a.tag}] {len(probs)} problems (kinds {sorted(kinds)}) x "
          f"{a.samples} samples = {len(seqs)} seqs; MAX_NEW={a.max_new} "
          f"TEMP={TEMP} seeds problems={a.seed} sample={a.sample_seed}",
          flush=True)

    g = torch.Generator().manual_seed(a.sample_seed)
    t_start = time.time()
    step_times = []
    total_new = 0
    for step in range(a.max_new):
        t0 = time.time()
        act = [r for r in seqs if not r["done"]]
        if not act:
            break
        logits = eng.forward_batch([r["ids"] for r in act])
        probs_cpu = torch.softmax(logits.float().cpu() / TEMP, -1)
        nxt = torch.multinomial(probs_cpu, 1, generator=g).squeeze(1).tolist()
        for r, t in zip(act, nxt):
            r["ids"].append(int(t))
            r["new"].append(int(t))
            if int(t) == eos:
                r["done"] = True
            total_new += 1
        dt = time.time() - t0
        step_times.append(dt)
        print(f"[{a.tag}] step {step:03d}: {len(act)} active, {dt:.1f}s "
              f"(eta {(sum(step_times)/len(step_times)) * (a.max_new - step - 1):.0f}s), "
              f"S_pack={sum(len(r['ids']) for r in act)}", flush=True)
    wall = time.time() - t_start

    rows = []
    by_pid = {}
    pmap = {p["pid"]: p for p in probs}
    for r in seqs:
        text = tok.decode(r["new"], skip_special_tokens=True)
        ans = pmap[r["pid"]]["ans"]
        got = parse_answer(text)
        fa = first_answer(text)
        rows.append({"pid": r["pid"], "kind": r["kind"],
                     "prompt": pmap[r["pid"]]["q"], "answer": ans,
                     "sample": r["s"], "budget": a.max_new,
                     "hit_cap": len(r["new"]) >= a.max_new,
                     "correct": got == ans, "correct_first_answer": fa == ans,
                     "parsed": got, "n_new": len(r["new"]), "completion": text})
        by_pid[r["pid"]] = by_pid.get(r["pid"], False) or (got == ans) or (fa == ans)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    per_kind = {}
    for k in sorted(kinds):
        rs = [r for r in rows if r["kind"] == k]
        okset = [r for r in rs if r["correct"] or r["correct_first_answer"]]
        per_kind[k] = {"samples": len(rs),
                       "correct_strict": sum(r["correct"] for r in rs),
                       "correct_first": sum(r["correct_first_answer"] for r in rs),
                       "hit_cap": sum(r["hit_cap"] for r in rs),
                       "mean_completion_tokens": round(
                           sum(r["n_new"] for r in rs) / max(len(rs), 1), 1),
                       "mean_completion_tokens_correct": round(
                           sum(r["n_new"] for r in okset) / max(len(okset), 1), 1)}
    report = {
        "tag": a.tag, "problems_seed": a.seed, "sample_seed": a.sample_seed,
        "kinds": sorted(kinds), "n_problems": len(probs),
        "n_samples": a.samples, "max_new": a.max_new, "temp": TEMP, "eos": eos,
        "correct_samples_strict": sum(r["correct"] for r in rows),
        "correct_samples_first": sum(r["correct_first_answer"] for r in rows),
        "total_samples": len(rows),
        "problems_with_correct": sum(by_pid.values()),
        "per_kind": per_kind,
        "wall_s": round(wall, 1), "step_times_s": [round(t, 1) for t in step_times],
        "tokens_generated": total_new,
        "tokens_per_s_aggregate": round(total_new / max(wall, 1), 2),
        "prompt_mode": "official chat template, non-thinking "
                       "(<bos><User>{q}<Assistant></think>)",
    }
    json.dump(report, open(a.out.replace(".jsonl", "_report.json"), "w"), indent=1)
    print(f"[{a.tag}] REPORT:", json.dumps({k: v for k, v in report.items()
                                            if k not in ("step_times_s", "per_kind")},
                                           indent=1), flush=True)
    print(f"[{a.tag}] per-kind:", json.dumps(per_kind), flush=True)
    print(f"[{a.tag}] sample completions:", flush=True)
    for row in rows[:6]:
        print(f"   pid{row['pid']} k{row['kind']} s{row['sample']} "
              f"ok={row['correct']} first={row['correct_first_answer']} "
              f"cap={row['hit_cap']} n={row['n_new']}: "
              f"{row['completion'][:90]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
