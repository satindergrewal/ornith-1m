#!/usr/bin/env python3
"""tc_token_ab.py - the ThinkingCap before/after TOKEN test (his order).

Bottlecap-style matched accounting: the SAME question goes to base and
capped; we measure completion tokens per question in both arms and the
per-question difference (matched delta %), never a raw mean alone (a
model that bails on hard questions would look efficient without being
so). Accuracy is reported beside tokens; the headline is token delta on
questions BOTH arms answered correctly.

Decoding follows the authors' protocol: temp 1.0, top_k 20 (top_p not
applied - the engine samples from the top-20 mask, which dominates).
Budget 1536 new tokens, 3 samples per problem, paired seeds.

Problems: 5 tiers, easy (the holdout class) through multi-step chains
and word problems - tiers chosen so the BASE model has something to
overthink (the holdout mean was 43 tokens; nothing to trim there).
"""

import argparse
import json
import random
import re
import time

import torch

from full_loader import StreamingDSV4
from dsv4_full import DSV4Full
from tc_batch_gen import build_prompt, parse_answer, first_answer

TEMP = 1.0
TOP_K = 20
MAX_NEW = 1536
N_SAMPLES = 3
PROBLEM_SEED = 20260990
SAMPLE_SEED = 20260991


def make_hard_problems(seed, n_per_tier=10):
    rng = random.Random(seed)
    probs = []
    pid = 0

    def add(kind, q, ans, tier):
        nonlocal pid
        probs.append({"pid": pid, "kind": kind, "tier": tier,
                      "q": q, "ans": ans})
        pid += 1

    for _ in range(n_per_tier):  # tier 0: holdout class (control)
        a, b, c = rng.randint(2, 49), rng.randint(2, 49), rng.randint(2, 9)
        add(2, f"What is {a} + {b} * {c}?", a + b * c, "t0-control")
    for _ in range(n_per_tier):  # tier 1: two products + precedence
        a, b, c, d = (rng.randint(6, 97) for _ in range(4))
        add(3, f"What is {a} * {b} + {c} * {d}?", a * b + c * d, "t1-twoproduct")
    for _ in range(n_per_tier):  # tier 2: three products, mixed signs
        a, b, c, d, e, f = (rng.randint(6, 89) for _ in range(6))
        add(3, f"What is {a} * {b} - {c} * {d} + {e} * {f}?",
            a * b - c * d + e * f, "t2-threeproduct")
    for _ in range(n_per_tier):  # tier 3: division requiring factoring
        b, k, c = rng.randint(6, 40), rng.randint(6, 40), rng.randint(2, 9)
        add(3, f"What is ({b * k} + {b}) / {b}?", k + 1, "t3-factor")
    for _ in range(n_per_tier):  # tier 4: word problem, 2-3 ops
        a, bb, c, d = rng.randint(6, 40), rng.randint(3, 19), \
            rng.randint(2, 30), rng.randint(10, 90)
        add(3, f"A warehouse ships {a} crates holding {bb} units each, "
               f"plus {c} extra boxes of {d} units. How many units in total?",
            a * bb + c * d, "t4-word")
    return probs


def run_arm(eng, tok, probs, arm):
    eos = tok.token_to_id("<|end|>") if hasattr(tok, "token_to_id") else None
    if eos is None:
        eos = eng.cfg.get("eos_token_id", 1) if hasattr(eng, "cfg") else 1
    seqs = []
    for p in probs:
        base = tok.encode(build_prompt(p["q"])).ids
        for s in range(N_SAMPLES):
            seqs.append({"pid": p["pid"], "tier": p["tier"], "s": s,
                         "ids": list(base), "done": False, "new": []})
    g = torch.Generator().manual_seed(SAMPLE_SEED)
    t0 = time.time()
    for step in range(MAX_NEW):
        act = [r for r in seqs if not r["done"]]
        if not act:
            break
        logits = eng.forward_batch([r["ids"] for r in act])
        lg = logits.float().cpu()
        if TOP_K and TOP_K < lg.shape[-1]:
            kth = lg.topk(TOP_K, dim=-1).values[:, -1:].expand_as(lg)
            lg = lg.masked_fill(lg < kth, float("-inf"))
        pc = torch.softmax(lg / TEMP, -1)
        nxt = torch.multinomial(pc, 1, generator=g).squeeze(1).tolist()
        for r, t in zip(act, nxt):
            r["ids"].append(int(t))
            r["new"].append(int(t))
            if int(t) == eos:
                r["done"] = True
        if step % 50 == 0:
            n_act = sum(1 for r in seqs if not r["done"])
            el = time.time() - t0
            print(f"[ab/{arm}] step {step:04d}: {n_act} active, "
                  f"{el:.0f}s elapsed", flush=True)
    rows = []
    for r in seqs:
        text = tok.decode(r["new"], skip_special_tokens=True)
        rows.append({"arm": arm, "pid": r["pid"], "tier": r["tier"],
                     "sample": r["s"],
                     "correct": parse_answer(text) ==
                     next(p["ans"] for p in probs if p["pid"] == r["pid"]),
                     "n_new": len(r["new"]),
                     "hit_cap": len(r["new"]) >= MAX_NEW,
                     "completion": text})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="/wd/cap/lora.safetensors")
    ap.add_argument("--n-per-tier", type=int, default=10)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("/model/tokenizer.json")
    probs = make_hard_problems(PROBLEM_SEED, args.n_per_tier)

    L = StreamingDSV4()
    eng = DSV4Full(L)

    out_all = []
    print(f"[ab] {len(probs)} problems x {N_SAMPLES} samples, "
          f"budget {MAX_NEW}, temp {TEMP} topk {TOP_K}", flush=True)

    base_rows = run_arm(eng, tok, probs, "base")
    out_all += base_rows
    json.dump(base_rows, open("/wd/data/token_ab_base.json", "w"), indent=1)

    from tc_lora_train import SCALE
    n = eng.attach_lora(args.lora, SCALE, fold=True)
    print(f"[ab] folded {n} adapters (scale {SCALE})", flush=True)
    cap_rows = run_arm(eng, tok, probs, "capped")
    out_all += cap_rows
    json.dump(cap_rows, open("/wd/data/token_ab_capped.json", "w"), indent=1)

    # ---- matched accounting ----
    def qmean(rows):
        m = {}
        for r in rows:
            m.setdefault(r["pid"], []).append(r)
        return {pid: sum(x["n_new"] for x in rs) / len(rs)
                for pid, rs in m.items()}, m

    bt, bm = qmean(base_rows)
    ct, cm = qmean(cap_rows)
    both_correct_pids = [p["pid"] for p in probs
                         if all(x["correct"] for x in bm[p["pid"]])
                         and all(x["correct"] for x in cm[p["pid"]])]
    deltas = [ct[pid] - bt[pid] for pid in both_correct_pids]
    sum_b = sum(bt[pid] for pid in both_correct_pids)
    sum_c = sum(ct[pid] for pid in both_correct_pids)
    acc_b = sum(1 for r in base_rows if r["correct"]) / len(base_rows)
    acc_c = sum(1 for r in cap_rows if r["correct"]) / len(cap_rows)
    mtb = sum(bt.values()) / len(bt)
    mtc = sum(ct.values()) / len(ct)

    tiers = sorted({p["tier"] for p in probs})
    per_tier = {}
    for t in tiers:
        pids = [p["pid"] for p in probs if p["tier"] == t]
        ok = [pid for pid in pids if pid in both_correct_pids]
        if ok:
            tb = sum(bt[i] for i in ok) / len(ok)
            tc_ = sum(ct[i] for i in ok) / len(ok)
            per_tier[t] = {"n": len(ok),
                           "base_tok": round(tb, 1),
                           "capped_tok": round(tc_, 1),
                           "delta_pct": round(100 * (tc_ - tb) / tb, 1)}
    summary = {
        "problems": len(probs), "samples": N_SAMPLES, "budget": MAX_NEW,
        "decoding": f"temp {TEMP} top_k {TOP_K}",
        "accuracy_base": round(acc_b, 3), "accuracy_capped": round(acc_c, 3),
        "mean_tokens_base": round(mtb, 1), "mean_tokens_capped": round(mtc, 1),
        "matched_n_questions": len(both_correct_pids),
        "matched_base_tokens": round(sum_b, 1),
        "matched_capped_tokens": round(sum_c, 1),
        "matched_delta_pct": round(100 * (sum_c - sum_b) / sum_b, 1),
        "hit_cap_base": sum(1 for r in base_rows if r["hit_cap"]),
        "hit_cap_capped": sum(1 for r in cap_rows if r["hit_cap"]),
        "per_tier": per_tier,
    }
    json.dump(summary, open("/wd/data/token_ab_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    with open("/wd/data/token_ab_rows.jsonl", "w") as f:
        for r in out_all:
            f.write(json.dumps(r) + "\n")
    print("[ab] DONE", flush=True)


if __name__ == "__main__":
    main()
