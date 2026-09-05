#!/usr/bin/env python3
"""glm_tc_batch_gen.py — ThinkingCap REAL on-policy regen for GLM-5.3-Flash
(port of tc_batch_gen.py; DSV4 harness is the template).

Prompt: official GLM 5.3 chat format from chat_template.jinja —
  [gMASK]<sop><|system|>Reasoning Effort: Low<|user|>{q}<|assistant|><think>
with </think> appended immediately (the DSV4 non-thinking equivalent: empty
think block -> direct answer). Special-token ids are composed
programmatically (token_to_id) because the tokenizer's text matching of
<|sop|> is unreliable; <|user|>/<|assistant|>/<think> DO match in text and
are encoded inline.

Stopping: any eos in text_config.eos_token_id. Rows packed into ONE forward
per step (forward_batch handles cu_seqlens row isolation for KDA and causal
per-row masking for DSA); temp 0.8 multinomial, seeded like the DSV4 chain.
"""

import argparse
import json
import os
import random
import re
import time

import torch

from glm_full_loader import StreamingGLM
from glm_full import GLMFull

TRAIN_SEED = 20260904
HOLDOUT_SEED = 20260905
GLOBAL_SEED = 20260977
TEMP = 0.8


def get_tok():
    from tokenizers import Tokenizer
    return Tokenizer.from_file("/root/glm-fp8/tokenizer.json")


def special_ids(tok):
    """GLM5 special tokens. NOTE the vocab names: '<sop>' (no pipes) =
    154824 — '<|sop|>' does NOT exist; 154823 is '[sMASK]'. [gMASK]/<sop>/
    <|system|>/<|user|>/<|assistant|>/<think>/</think> verified via
    id_to_token on the FP8 tokenizer."""
    ids = {}
    for s in ("[gMASK]", "<sop>", "<|system|>", "<|user|>",
              "<|assistant|>", "<think>", "</think>"):
        i = tok.token_to_id(s)
        if i is None:
            i = {"[gMASK]": 154822, "<sop>": 154824, "<|system|>": 154826,
                 "<|user|>": 154827, "<|assistant|>": 154828,
                 "<think>": 154841, "</think>": 154842}[s]
        ids[s] = int(i)
    return ids


def build_prompt_ids(tok, sp, q):
    """[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>q<|assistant|><think></think>"""
    seg = tok.encode("<|system|>Reasoning Effort: Low<|user|>",
                     add_special_tokens=False).ids
    # <|system|>/<|user|> match as added tokens in text (verified); fall back
    # to programmatic composition if the tokenizer ever stops matching them.
    if sp["<|system|>"] not in seg[:1]:
        seg = ([sp["<|system|>"]]
               + tok.encode("Reasoning Effort: Low", add_special_tokens=False).ids
               + [sp["<|user|>"]])
    else:
        seg = [sp["<|system|>"]]
        seg += tok.encode("Reasoning Effort: Low", add_special_tokens=False).ids
        seg += [sp["<|user|>"]]
    return ([sp["[gMASK]"], sp["<sop>"]] + seg
            + tok.encode(q, add_special_tokens=False).ids
            + [sp["<|assistant|>"], sp["<think>"], sp["</think>"]])


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
    ap.add_argument("--out", type=str, default="/root/tc-glm/data/regen.jsonl")
    ap.add_argument("--tag", type=str, default="regen")
    ap.add_argument("--problems", type=str, default=None,
                    help="JSONL problem file (rows: pid,kind,q,ans)")
    a = ap.parse_args()
    kinds = {int(k) for k in a.kinds.split(",")}

    tok = get_tok()
    sp = special_ids(tok)

    if a.problems:
        probs = [p for p in (json.loads(l) for l in open(a.problems))
                 if p["kind"] in kinds]
        print(f"[{a.tag}] loaded {len(probs)} problems from {a.problems}",
              flush=True)
    else:
        probs = [p for p in make_problems(a.seed, a.n) if p["kind"] in kinds]

    L = StreamingGLM()
    eng = GLMFull(L)
    eos_set = set(eng.eos)

    seqs = []
    for p in probs:
        base = build_prompt_ids(tok, sp, p["q"])
        for s in range(a.samples):
            seqs.append({"pid": p["pid"], "kind": p["kind"], "s": s,
                         "ids": list(base), "done": False, "new": []})
    print(f"[{a.tag}] {len(probs)} problems (kinds {sorted(kinds)}) x "
          f"{a.samples} samples = {len(seqs)} seqs; MAX_NEW={a.max_new} "
          f"TEMP={TEMP} seeds problems={a.seed} sample={a.sample_seed} "
          f"eos={sorted(eos_set)}", flush=True)

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
            if int(t) in eos_set:
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
        "n_samples": a.samples, "max_new": a.max_new, "temp": TEMP,
        "eos": sorted(eos_set),
        "correct_samples_strict": sum(r["correct"] for r in rows),
        "correct_samples_first": sum(r["correct_first_answer"] for r in rows),
        "total_samples": len(rows),
        "problems_with_correct": sum(by_pid.values()),
        "per_kind": per_kind,
        "wall_s": round(wall, 1), "step_times_s": [round(t, 1) for t in step_times],
        "tokens_generated": total_new,
        "tokens_per_s_aggregate": round(total_new / max(wall, 1), 2),
        "prompt_mode": "official GLM chat template, low effort + closed think",
    }
    json.dump(report, open(a.out.replace(".jsonl", "_report.json"), "w"), indent=1)
    print(f"[{a.tag}] REPORT:", json.dumps(
        {k: v for k, v in report.items() if k not in ("step_times_s",)},
        indent=None)[:600], flush=True)


if __name__ == "__main__":
    main()
