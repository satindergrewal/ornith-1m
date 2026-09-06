#!/usr/bin/env python3
"""dsv4_source_parity.py - the gate: CappedSource == fold_cap_to_bf16 bitwise.

The fold wrote 6 complete shards (with .done markers) before the volume
quota killed it. Those shards are the oracle: every tensor in each folded
shard must be BITWISE equal to what CappedSource produces for the same
name. Any mismatch means the dequant-on-demand path is not the fold, and
the encode must not run off it.

Comparison is exact (== for all elements, NaN==NaN tolerated). The fold and
this source run the same LUTs, same fp32 accumulate, same single bf16
downcast, on the same GPU class - bitwise equality is the expected result,
not a hope.
"""

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

from dsv4_capped_source import CappedSource

from dsv4_geometry import ROUTED, MTP_ROUTED


def name_to_tensor(src, name):
    if ROUTED.match(name) or MTP_ROUTED.match(name):
        return src.expert_tensor(name)
    return src.load_native(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="/model")
    ap.add_argument("--folded-dir", default="/workspace/model-capped-bf16")
    ap.add_argument("--lora", default="/wd/cap/lora.safetensors")
    args = ap.parse_args()

    src = CappedSource(model_dir=args.src_model, lora_path=args.lora)

    shards = sorted(f for f in os.listdir(args.folded_dir)
                    if f.endswith(".safetensors")
                    and os.path.exists(os.path.join(
                        args.folded_dir, f + ".done")))
    if not shards:
        sys.exit("[parity] no folded shards with .done markers - no oracle")
    print(f"[parity] oracle shards: {len(shards)}", flush=True)

    n_eq = n_bad = n_missing = 0
    failures = []
    for sf in shards:
        with safe_open(os.path.join(args.folded_dir, sf),
                       framework="pt") as f:
            names = list(f.keys())
            for i, name in enumerate(names):
                if name not in src.L.meta:
                    n_missing += 1
                    failures.append(f"missing-in-meta {name}")
                    continue
                want = f.get_tensor(name)
                got = name_to_tensor(src, name).to(want.device)
                if got.dtype != want.dtype:
                    n_bad += 1
                    failures.append(f"dtype {name}: {got.dtype} vs {want.dtype}")
                    continue
                if got.shape != want.shape:
                    n_bad += 1
                    failures.append(f"shape {name}: {got.shape} vs {want.shape}")
                    continue
                eq = (got == want) | (torch.isnan(got.float())
                                      & torch.isnan(want.float()))
                if bool(eq.all()):
                    n_eq += 1
                else:
                    n_bad += 1
                    diff = (got.float() - want.float()).abs()
                    failures.append(
                        f"bits {name}: {(~eq).sum().item()} elems, "
                        f"max|d| {diff.max().item():.6f}")
                if (i + 1) % 200 == 0:
                    print(f"[parity] {sf}: {i+1}/{len(names)} "
                          f"(eq {n_eq} bad {n_bad})", flush=True)
        print(f"[parity] {sf} DONE: {len(names)} tensors "
              f"(running eq {n_eq} bad {n_bad} miss {n_missing})", flush=True)

    verdict = {"shards": len(shards), "equal": n_eq, "mismatched": n_bad,
               "missing": n_missing,
               "status": "PASS" if (n_bad == 0 and n_missing == 0) else "FAIL",
               "failures_head": failures[:10]}
    out = os.path.join(args.folded_dir, "SOURCE_PARITY.json")
    json.dump(verdict, open(out, "w"), indent=1)
    print(json.dumps(verdict, indent=1), flush=True)
    sys.exit(0 if verdict["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
