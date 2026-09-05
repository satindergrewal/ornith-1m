#!/usr/bin/env python3
"""glm_tc_holdout_gate.py — before/after gate for the GLM-5.3-Flash cap
(port of tc_holdout_gate.py). Holdout seed 20260905 (12 problems, first
touched here, never trained on). base arm vs capped arm (LoRA folded into
the loaded bf16 weights, W_eff = W + (alpha/r) B@A); 3 samples/problem,
temp 0.8, budget 160, SAME sample seed (paired). Official GLM chat prompt
(low effort + closed think)."""

import argparse
import json
import time

import torch

from glm_full_loader import StreamingGLM
from glm_full import GLMFull
from glm_tc_batch_gen import (get_tok, special_ids, build_prompt_ids,
                              make_problems, parse_answer, first_answer)

HOLDOUT_SEED = 20260905
GATE_SAMPLE_SEED = 20260978
TEMP = 0.8
MAX_NEW = 160
N_SAMPLES = 3
DATA = "/root/tc-glm/data"


def run_arm(arm, lora_path=None):
    tok = get_tok()
    sp = special_ids(tok)
    L = StreamingGLM()
    eng = GLMFull(L)
    eos_set = set(eng.eos)
    if arm == "capped":
        from glm_tc_lora_train import SCALE
        n = eng.attach_lora(lora_path, SCALE, fold=True)
        print(f"[gate/{arm}] folded {n} adapters (scale {SCALE})", flush=True)

    probs = make_problems(HOLDOUT_SEED, 12)
    seqs = []
    for p in probs:
        base = build_prompt_ids(tok, sp, p["q"])
        for s in range(N_SAMPLES):
            seqs.append({"pid": p["pid"], "kind": p["kind"], "s": s,
                         "ids": list(base), "done": False, "new": []})
    g = torch.Generator().manual_seed(GATE_SAMPLE_SEED)
    t0 = time.time()
    step_times = []
    for step in range(MAX_NEW):
        ts = time.time()
        act = [r for r in seqs if not r["done"]]
        if not act:
            break
        logits = eng.forward_batch([r["ids"] for r in act])
        pc = torch.softmax(logits.float().cpu() / TEMP, -1)
        nxt = torch.multinomial(pc, 1, generator=g).squeeze(1).tolist()
        for r, t in zip(act, nxt):
            r["ids"].append(int(t))
            r["new"].append(int(t))
            if int(t) in eos_set:
                r["done"] = True
        step_times.append(time.time() - ts)
        if step % 10 == 0:
            print(f"[gate/{arm}] step {step:03d}: {len(act)} active, "
                  f"{step_times[-1]:.1f}s", flush=True)
    wall = time.time() - t0

    pmap = {p["pid"]: p for p in probs}
    rows = []
    for r in seqs:
        text = tok.decode(r["new"], skip_special_tokens=True)
        ans = pmap[r["pid"]]["ans"]
        rows.append({"arm": arm, "pid": r["pid"], "kind": r["kind"],
                     "sample": r["s"], "answer": ans,
                     "correct": parse_answer(text) == ans,
                     "correct_first_answer": first_answer(text) == ans,
                     "n_new": len(r["new"]),
                     "hit_cap": len(r["new"]) >= MAX_NEW,
                     "completion": text})
    with open(f"{DATA}/holdout_{arm}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def summarize(rs):
        ok = [r for r in rs if r["correct"] or r["correct_first_answer"]]
        return {"samples": len(rs),
                "correct_strict": sum(r["correct"] for r in rs),
                "correct_first": sum(r["correct_first_answer"] for r in rs),
                "hit_cap": sum(r["hit_cap"] for r in rs),
                "mean_tokens_all": round(
                    sum(r["n_new"] for r in rs) / max(len(rs), 1), 1),
                "mean_tokens_correct": round(
                    sum(r["n_new"] for r in ok) / max(len(ok), 1), 1)}

    rep = {"arm": arm, "lora": lora_path, "seed": HOLDOUT_SEED,
           "sample_seed": GATE_SAMPLE_SEED, "temp": TEMP, "max_new": MAX_NEW,
           "wall_s": round(wall, 1),
           "overall": summarize(rows),
           "per_kind": {str(k): summarize([r for r in rows if r["kind"] == k])
                        for k in (0, 1, 2)}}
    json.dump(rep, open(f"{DATA}/holdout_{arm}_report.json", "w"), indent=1)
    print(f"[gate/{arm}] REPORT: {json.dumps(rep['overall'])}", flush=True)
    print(f"[gate/{arm}] per-kind: {json.dumps(rep['per_kind'])}", flush=True)
    for r in rows[:6]:
        print(f"   pid{r['pid']} k{r['kind']} ok={r['correct']} "
              f"n={r['n_new']}: {r['completion'][:70]!r}", flush=True)
    return rep


def compare():
    b = json.load(open(f"{DATA}/holdout_base_report.json"))
    c = json.load(open(f"{DATA}/holdout_capped_report.json"))
    print("=== HOLDOUT GATE: base vs capped ===")
    for name, arm in (("base", b), ("capped", c)):
        print(f"{name:6s} overall {json.dumps(arm['overall'])}")
        for k, v in arm["per_kind"].items():
            print(f"       kind{k} {json.dumps(v)}")
    ob, oc = b["overall"], c["overall"]
    d_acc = oc["correct_first"] - ob["correct_first"]
    d_len = oc["mean_tokens_correct"] - ob["mean_tokens_correct"]
    print(f"delta: correct_first {d_acc:+d}/{ob['samples']}, "
          f"mean_tokens_correct {d_len:+.1f}")
    print(f"GATE accuracy-preserved: {'PASS' if d_acc >= -2 else 'CHECK'} "
          f"(within noise: 2 samples of {ob['samples']})")
    print(f"GATE length-bounded: {'PASS' if d_len < 0 else 'FAIL'} "
          f"(capped mean tokens {oc['mean_tokens_correct']} vs base "
          f"{ob['mean_tokens_correct']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "capped", "compare"])
    ap.add_argument("--lora", default="/root/tc-glm/cap/lora.safetensors")
    a = ap.parse_args()
    if a.cmd == "compare":
        compare()
    else:
        run_arm(a.cmd, a.lora if a.cmd == "capped" else None)
