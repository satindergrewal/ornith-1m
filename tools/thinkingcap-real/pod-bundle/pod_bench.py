#!/usr/bin/env python3
"""Pod-scale ThinkingCap A/B bench: base vs live-fp32 cap on the REAL
DSV4-Flash-Vision ablit, thinking-ON (no prefilled </think>), greedy.

Answers his question with numbers: does the cap reach the same answers
in fewer tokens?

Run inside the container: /model = checkpoint, /wd = bundle dir.
One process per arm; wrapper pins CUDA_VISIBLE_DEVICES per process.
"""
import argparse
import json
import os
import random
import re
import time

import torch
from tokenizers import Tokenizer

from full_loader import StreamingDSV4
from dsv4_full import DSV4Full
from tc_lora_train import SCALE

BOS = "<｜begin▁of▁sentence｜>"
USER_SP = "<｜User｜>"
ASSISTANT_SP = "<｜Assistant｜>"

OPEN_PROMPTS = [
    "can you make quick sort in C and assembly? and tell me which is "
    "the smallest code and fastest and why?",
    "A train leaves at 3pm travelling 60 km/h; another leaves an hour "
    "later at 90 km/h on the same track. At what time does the second "
    "catch the first? Show your reasoning, then answer.",
]


def build_prompt_thinking(q):
    return BOS + USER_SP + q + ASSISTANT_SP


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


def run_batch(eng, tok, eos, seqs, budget, tag):
    """seqs: list of dicts with ids; greedy batched generation."""
    t_start = time.time()
    step_times = []
    for step in range(budget):
        act = [r for r in seqs if not r["done"]]
        if not act:
            break
        t0 = time.time()
        logits = eng.forward_batch([r["ids"] for r in act])
        nxt = logits.argmax(-1).tolist()
        for r, t in zip(act, nxt):
            r["ids"].append(int(t))
            r["new"].append(int(t))
            if int(t) == eos:
                r["done"] = True
        dt = time.time() - t0
        step_times.append(dt)
        if step % 16 == 0 or step == budget - 1:
            eta = (sum(step_times) / len(step_times)) * (budget - step - 1)
            print(f"[{tag}] step {step:03d}: {len(act)} active, {dt:.1f}s "
                  f"(eta {eta:.0f}s)", flush=True)
    return time.time() - t_start


def summarize(rows):
    out = {}
    for kind in sorted(set(r["kind"] for r in rows)):
        rs = [r for r in rows if r["kind"] == kind]
        corr = [r for r in rs if r["correct"]]
        out[f"kind{kind}"] = {
            "n": len(rs),
            "correct": sum(1 for r in rs if r["correct"]),
            "mean_tokens_all": round(sum(r["n_new"] for r in rs) / len(rs), 1),
            "mean_tokens_correct":
                round(sum(r["n_new"] for r in corr) / len(corr), 1) if corr else None,
            "hit_cap": sum(1 for r in rs if r["hit_cap"]),
        }
    corr = [r for r in rows if r["correct"]]
    out["total"] = {
        "n": len(rows),
        "correct": len(corr),
        "mean_tokens_all": round(sum(r["n_new"] for r in rows) / len(rows), 1),
        "mean_tokens_correct":
            round(sum(r["n_new"] for r in corr) / len(corr), 1) if corr else None,
        "hit_cap": sum(1 for r in rows if r["hit_cap"]),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["base", "cap"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--budget", type=int, default=768)
    ap.add_argument("--open-budget", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = Tokenizer.from_file("/model/tokenizer.json")
    eng = DSV4Full(StreamingDSV4(model_dir="/model",
                                 meta_path="/wd/tensor_meta.json"))
    eos = eng.eos

    if a.arm == "cap":
        n = eng.attach_lora("/wd/lora.safetensors", SCALE, fold=False)
        print(f"[cap] attached {n} live fp32 adapters (scale {SCALE})",
              flush=True)

    # phase 1: math problems, thinking-on
    probs = make_problems(a.seed, a.n)
    seqs = []
    for p in probs:
        ids = tok.encode(build_prompt_thinking(p["q"])).ids
        seqs.append({"pid": p["pid"], "kind": p["kind"], "ids": list(ids),
                     "new": [], "done": False})
    wall = run_batch(eng, tok, eos, seqs, a.budget, f"{a.arm}-math")
    rows = []
    for r, p in zip(seqs, probs):
        text = tok.decode(r["new"], skip_special_tokens=True)
        got = parse_answer(text)
        rows.append({"pid": r["pid"], "kind": r["kind"], "q": p["q"],
                     "answer": p["ans"], "parsed": got,
                     "correct": got == p["ans"],
                     "n_new": len(r["new"]),
                     "hit_cap": len(r["new"]) >= a.budget,
                     "completion": text[-1500:]})

    # phase 2: open prompts (qualitative)
    oseqs = []
    for i, q in enumerate(OPEN_PROMPTS):
        ids = tok.encode(build_prompt_thinking(q)).ids
        oseqs.append({"pid": i, "ids": list(ids), "new": [], "done": False})
    run_batch(eng, tok, eos, oseqs, a.open_budget, f"{a.arm}-open")
    open_rows = []
    for r, q in zip(oseqs, OPEN_PROMPTS):
        text = tok.decode(r["new"], skip_special_tokens=True)
        open_rows.append({"q": q, "n_new": len(r["new"]),
                          "hit_cap": len(r["new"]) >= a.open_budget,
                          "completion": text[-2500:]})

    report = {"arm": a.arm, "seed": a.seed, "budget": a.budget,
              "open_budget": a.open_budget, "scale": SCALE,
              "wall_s": round(wall, 1), "math": summarize(rows),
              "math_rows": rows, "open": open_rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[{a.arm}] MATH: {json.dumps(report['math'])}", flush=True)
    for o in open_rows:
        print(f"[{a.arm}] open n_new={o['n_new']} hit_cap={o['hit_cap']}",
              flush=True)
    print(f"[{a.arm}] DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
