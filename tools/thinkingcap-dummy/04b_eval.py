#!/usr/bin/env python3
"""ThinkingCap dummy - STEP 04b: pre-merge LoRA eval (his call).

Apply the trained adapters from cap/lora.safetensors in memory
(W_eff = W + (alpha/r)*(B@A)) - mathematically identical to the merged
checkpoint - and compare base vs capped on HELD-OUT problems (fresh
seed, disjoint from the SFT set). Reports answer correctness and
completion length: the cap's product metric is shorter completions at
equal-or-better correctness.

Runs on CPU. Output: <wd>/cap/eval_report.json
"""
import json
import os
import random
import re
import sys
from pathlib import Path

import torch

WD = Path(os.environ.get("TC_WD", "/home/satinder/thinkingcap-dummy"))
sys.path.insert(0, str(WD))
import nano_torch as nt  # noqa

CAP = WD / "cap"
SFT_SEED_PROBLEMS = 20260903  # 03_regen_data.py seed
HOLDOUT_SEED = 20260904
R = 8
ALPHA = 16
SCALE = ALPHA / R


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
        probs.append((q, ans))
    return probs


def parse_answer(text):
    m = re.findall(r"-?\d+", text)
    return int(m[-1]) if m else None


def apply_adapters(model):
    from safetensors.torch import load_file
    tensors = load_file(str(CAP / "lora.safetensors"))
    patched = 0
    for i, L in enumerate(model.M.layers):
        A_at = L.attn
        for attr, raw in (("wq_b", f"layers.{i}.attn.wq_b.weight"),
                          ("wkv", f"layers.{i}.attn.wkv.weight"),
                          ("wo_b", f"layers.{i}.attn.wo_b.weight")):
            A = tensors[f"lora.{raw}.A"]
            B = tensors[f"lora.{raw}.B"]
            setattr(A_at, attr, getattr(A_at, attr).detach()
                    + SCALE * (B @ A))
            patched += 1
        for attr, name in (("sw1", "w1"), ("sw2", "w2"), ("sw3", "w3")):
            raw = f"layers.{i}.ffn.shared_experts.{name}.weight"
            A = tensors[f"lora.{raw}.A"]
            B = tensors[f"lora.{raw}.B"]
            setattr(L.ffn, attr, getattr(L.ffn, attr).detach()
                    + SCALE * (B @ A))
            patched += 1
    return patched


def main():
    torch.manual_seed(HOLDOUT_SEED)
    tok = nt.get_tokenizer()
    probs = holdout_problems()

    model = nt.NanoDSV4()
    results = {"base": [], "cap": []}
    for q, ans in probs:
        ids = tok.encode(q).ids
        out = model.generate(ids, max_new_tokens=48)
        txt = tok.decode(out[len(ids):], skip_special_tokens=True)
        results["base"].append({"q": q, "gold": ans,
                                "pred": parse_answer(txt),
                                "len": len(out) - len(ids)})

    n = apply_adapters(model)
    for q, ans in probs:
        ids = tok.encode(q).ids
        out = model.generate(ids, max_new_tokens=48)
        txt = tok.decode(out[len(ids):], skip_special_tokens=True)
        results["cap"].append({"q": q, "gold": ans,
                               "pred": parse_answer(txt),
                               "len": len(out) - len(ids)})

    def stats(rows):
        correct = sum(1 for r in rows if r["pred"] == r["gold"])
        return {"correct": correct, "n": len(rows),
                "mean_completion_len":
                    round(sum(r["len"] for r in rows) / len(rows), 2)}

    report = {"adapters_patched": n,
              "base": stats(results["base"]),
              "cap": stats(results["cap"]),
              "off_policy_training": True,
              "note": "dummy: 03 regen fell back 100% to teacher "
                      "solutions, so the cap rehearse the mechanism, "
                      "not a quality win"}
    (CAP / "eval_report.json").write_text(json.dumps(report, indent=1))
    print("EVAL:", json.dumps(report))


if __name__ == "__main__":
    main()
