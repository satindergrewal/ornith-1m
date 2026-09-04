#!/usr/bin/env python3
"""ThinkingCap dummy - STEP 06: before/after generation A/B (his call).

Base (nano-dsv4-vision-ablit) vs the MERGED cap artifact
(nano-dsv4-vision-ablit-cap) - the actual product of 01b->05, not an
in-memory adapter. Greedy, deterministic. Answers for the dummy:
  1. does our cap change generation AT ALL (token-level divergence)?
  2. where does it first diverge?
  3. completion length before vs after (the product metric)?
  4. correctness preserved?
04b used a 48-token budget - every completion truncated, no signal.
This run uses 160+ per the banked gotcha.

Runs on CPU (LilMonkey venv). Output: <wd>/cap/ab_bench.json
"""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch

WD = Path(os.environ.get("TC_WD", "/home/satinder/thinkingcap-dummy"))
sys.path.insert(0, str(WD))
import nano_torch as nt  # noqa

BASE_DIR = WD / "nano-dsv4-vision-ablit"
CAP_DIR = WD / "nano-dsv4-vision-ablit-cap"
HOLDOUT_SEED = 20260904  # same as 04b: comparable problems
MATH_BUDGET = 160
OPEN_BUDGET = 256

OPEN_PROMPTS = [
    "can you make quick sort in C and assembly? and tell me which is "
    "the smallest code and fastest and why?",
    "Explain in one paragraph why the sky is blue.",
]


def holdout_problems(n=30):
    rng = random.Random(HOLDOUT_SEED)
    probs = []
    for i in range(n):
        a, b, c = rng.randint(50, 99), rng.randint(50, 99), rng.randint(10, 19)
        kind = i % 3
        if kind == 0:
            q, ans = f"What is {a} + {b}?", a + b
        elif kind == 1:
            q, ans = f"What is {a} * {c}?", a * c
        else:
            q, ans = f"What is {a} + {b} * {c}?", a + b * c
        probs.append({"q": q, "gold": ans, "kind": kind})
    return probs


def parse_answer(text):
    m = re.findall(r"-?\d+", text)
    return int(m[-1]) if m else None


def run_arm(model, tok, q, budget):
    ids = tok.encode(q).ids
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=budget)
    gen = out[len(ids):]
    txt = tok.decode(gen, skip_special_tokens=True)
    return {"gen": gen, "txt": txt, "len": len(gen),
            "eos": gen[-1] == model.M.eos if gen else False,
            "sec": round(time.time() - t0, 1)}


def main():
    torch.manual_seed(HOLDOUT_SEED)
    tok = nt.get_tokenizer()
    base = nt.NanoDSV4(str(BASE_DIR))
    cap = nt.NanoDSV4(str(CAP_DIR))

    # logit-level tie-in: same-shape check as the merge receipt
    ids = tok.encode("What is 7 + 5?").ids
    with torch.no_grad():
        d = float((cap.forward(ids) - base.forward(ids)).abs().max())
    print(f"probe maxdiff base vs merged-cap logits: {d}", flush=True)

    rows = []
    for i, p in enumerate(holdout_problems()):
        b = run_arm(base, tok, p["q"], MATH_BUDGET)
        c = run_arm(cap, tok, p["q"], MATH_BUDGET)
        div = next((j for j, (x, y) in enumerate(zip(b["gen"], c["gen"]))
                    if x != y), None)
        rows.append({"q": p["q"], "kind": p["kind"], "gold": p["gold"],
                     "base": {"len": b["len"], "eos": b["eos"],
                              "pred": parse_answer(b["txt"])},
                     "cap": {"len": c["len"], "eos": c["eos"],
                             "pred": parse_answer(c["txt"])},
                     "first_div": div})
        print(f"[{i+1}/30] kind{p['kind']} base {b['len']}tok "
              f"eos={b['eos']} | cap {c['len']}tok eos={c['eos']} | "
              f"div@{div}", flush=True)

    open_rows = []
    for q in OPEN_PROMPTS:
        b = run_arm(base, tok, q, OPEN_BUDGET)
        c = run_arm(cap, tok, q, OPEN_BUDGET)
        div = next((j for j, (x, y) in enumerate(zip(b["gen"], c["gen"]))
                    if x != y), None)
        open_rows.append({"q": q, "base": b, "cap": c, "first_div": div,
                          "base_txt": b["txt"][:600],
                          "cap_txt": c["txt"][:600]})
        print(f"[open] base {b['len']}tok | cap {c['len']}tok | div@{div}",
              flush=True)

    def agg(items, arm):
        lens = [r[arm]["len"] for r in items]
        corr = [r[arm]["pred"] == r["gold"] for r in items]
        return {"mean_len": round(sum(lens) / len(lens), 1),
                "correct": sum(corr), "n": len(items)}

    math_summary = {"base": agg(rows, "base"), "cap": agg(rows, "cap"),
                    "diverged": sum(1 for r in rows if r["first_div"] is not None),
                    "identical_streams": sum(1 for r in rows
                                             if r["first_div"] is None),
                    "mean_first_div": round(sum(r["first_div"] for r in rows
                                                if r["first_div"] is not None)
                                            / max(1, sum(1 for r in rows
                                                         if r["first_div"] is not None)), 1),
                    "base_eos": sum(1 for r in rows if r["base"]["eos"]),
                    "cap_eos": sum(1 for r in rows if r["cap"]["eos"])}

    report = {"probe_maxdiff": d, "budget_math": MATH_BUDGET,
              "budget_open": OPEN_BUDGET, "math": math_summary,
              "math_rows": rows, "open": open_rows}
    (WD / "cap" / "ab_bench.json").write_text(json.dumps(report, indent=1))
    print("MATH:", json.dumps(math_summary), flush=True)
    print("DONE -> cap/ab_bench.json", flush=True)


if __name__ == "__main__":
    main()
