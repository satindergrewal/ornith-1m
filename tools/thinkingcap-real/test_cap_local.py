#!/usr/bin/env python3
"""test_cap_local.py - A/B runner: DSV4 Flash ablit base vs capped, on
the packed master + the cap LoRA keeper. No merge, no EXL3.

The engine has no KV cache (full recompute per token), so this batches
ALL sequences - every question x every arm - into one forward_batch per
step, the regen pattern: the per-step expert-sweep cost is amortized
across the whole batch. 10 questions x 2 arms finishes in roughly the
time 1 single-stream question would take naive.

  scripted:     python3 test_cap_local.py --lora lora.safetensors --file prompts.txt
  interactive:  python3 test_cap_local.py --lora lora.safetensors
                (enter questions, blank line to run, empty to quit)
  cap-only:     --cap on|off|both   (default both, side-by-side)
"""

import argparse
import os
import sys
import time

import torch

BUNDLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "pod-bundle")
sys.path.insert(0, BUNDLE)

from full_loader import StreamingDSV4  # noqa: E402
from dsv4_full import DSV4Full  # noqa: E402
from tc_batch_gen import build_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get(
        "DSV4_MODEL", "/mnt/t5evo/dsv4-vision-ablit"))
    ap.add_argument("--meta", default=None,
                    help="tensor_meta.json (default <model-dir>/tensor_meta.json)")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--cap", choices=("on", "off", "both"), default="both")
    ap.add_argument("--file", default=None)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    meta = args.meta or os.path.join(args.model_dir, "tensor_meta.json")

    from tokenizers import Tokenizer
    from tc_lora_train import SCALE

    tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    eos = tok.token_to_id("<|end|>")

    if args.file:
        questions = [l.strip() for l in open(args.file) if l.strip()]
    else:
        questions = []
        print("[test] enter questions, blank line to run:", flush=True)
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                if questions:
                    break
                continue
            questions.append(line)
    if not questions:
        sys.exit("[test] no questions")

    arms = (("base", False), ("capped", True)) if args.cap == "both" \
        else (("capped", True) if args.cap == "on" else (("base", False),))

    results = {}
    for arm_name, capped in arms:
        eng = DSV4Full(StreamingDSV4(args.model_dir, meta))
        if capped:
            n = eng.attach_lora(args.lora, SCALE, fold=True)
            print(f"[test] {arm_name}: folded {n} adapters", flush=True)
        seqs = [{"q": q, "ids": list(tok.encode(build_prompt(q)).ids),
                 "new": [], "done": False} for q in questions]
        g = torch.Generator().manual_seed(20260990)
        t0 = time.time()
        for step in range(args.max_new):
            act = [s for s in seqs if not s["done"]]
            if not act:
                break
            logits = eng.forward_batch([s["ids"] for s in act])
            lg = logits.float().cpu()
            if args.top_k and args.top_k < lg.shape[-1]:
                kth = lg.topk(args.top_k, dim=-1).values[:, -1:]
                lg = lg.masked_fill(lg < kth, float("-inf"))
            probs = torch.softmax(lg / args.temp, -1)
            nxt = torch.multinomial(probs, 1, generator=g).squeeze(1).tolist()
            for s, t in zip(act, nxt):
                s["ids"].append(int(t))
                s["new"].append(int(t))
                if int(t) == eos:
                    s["done"] = True
            if step % 25 == 0:
                alive = sum(1 for s in seqs if not s["done"])
                print(f"[test] {arm_name} step {step}: {alive} active, "
                      f"{time.time()-t0:.0f}s", flush=True)
        for q, s in zip(questions, seqs):
            results.setdefault(q, {})[arm_name] = tok.decode(
                s["new"], skip_special_tokens=True)
        del eng
        torch.cuda.empty_cache()
        print(f"[test] {arm_name} arm done in {time.time()-t0:.0f}s",
              flush=True)

    for q, arms_out in results.items():
        print(f"\n{'='*70}\nQ: {q}")
        for arm_name, _ in arms:
            if arm_name in arms_out:
                words = len(arms_out[arm_name].split())
                print(f"\n--- {arm_name} ({words} words):\n{arms_out[arm_name]}")
    print(flush=True)


if __name__ == "__main__":
    main()
