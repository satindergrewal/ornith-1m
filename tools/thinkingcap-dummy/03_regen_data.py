#!/usr/bin/env python3
"""ThinkingCap dummy - STEP 03: on-policy regen (hotdogs method).

Grade-school arithmetic problems; 8 sampled completions per problem
from the CURRENT base (nano ablit via nano_torch), oracle filter,
shortest-correct -> SFT set, shortest-vs-longest correct -> DPO pairs.
If the base produces too few correct samples (it is a dummy), top up
with templated teacher solutions and FLAG them off-policy in the
report - the mechanism is the product, not the dummy's quality.

Runs on CPU. Uses nano_torch.py as the inference engine (import).
Output: <wd>/regen/{sft.jsonl,dpo.jsonl,report.json}
"""
import json
import os
import random
import re
import sys
from pathlib import Path

WD = Path(os.environ.get("TC_WD", "/home/satinder/thinkingcap-dummy"))
sys.path.insert(0, str(WD))
from nano_torch import load_model, generate_text, tokenize, detokenize  # noqa

OUT = WD / "regen"
N_PROBLEMS = 200
N_SAMPLES = 8
MIN_CORRECT_PROBLEMS = 30


def make_problems(seed=20260903):
    rng = random.Random(seed)
    probs = []
    for i in range(N_PROBLEMS):
        a, b, c = rng.randint(2, 49), rng.randint(2, 49), rng.randint(2, 9)
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


def main():
    model = load_model(str(WD / "nano-dsv4-vision-ablit"))
    probs = make_problems()
    sft, dpo, stats = [], [], {"correct_samples": 0, "problems_with_correct": 0}
    for q, ans in probs:
        correct, wrong = [], []
        for s in range(N_SAMPLES):
            txt = generate_text(model, q, max_new_tokens=64,
                                temperature=0.8, seed=s)
            got = parse_answer(txt)
            if got == ans:
                correct.append(txt)
            else:
                wrong.append(txt)
        stats["correct_samples"] += len(correct)
        if correct:
            stats["problems_with_correct"] += 1
            short = min(correct, key=len)
            sft.append({"prompt": q, "completion": short})
            if len(correct) > 1:
                long_ = max(correct, key=len)
                if long_ != short:
                    dpo.append({"prompt": q, "chosen": short,
                                "rejected": long_})
            elif wrong:
                dpo.append({"prompt": q, "chosen": short,
                            "rejected": wrong[0]})
    off_policy = 0
    if stats["problems_with_correct"] < MIN_CORRECT_PROBLEMS:
        rng = random.Random(7)
        for q, ans in probs:
            if len(sft) >= MIN_CORRECT_PROBLEMS:
                break
            if not any(r["prompt"] == q for r in sft):
                sft.append({"prompt": q,
                            "completion": f" {ans}",
                            "off_policy": True})
                off_policy += 1
    OUT.mkdir(exist_ok=True)
    (OUT / "sft.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in sft))
    (OUT / "dpo.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in dpo))
    report = {**stats, "sft": len(sft), "dpo": len(dpo),
              "off_policy_topup": off_policy}
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print("REGEN:", json.dumps(report))


if __name__ == "__main__":
    main()
