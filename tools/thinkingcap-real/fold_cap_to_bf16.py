#!/usr/bin/env python3
"""fold_cap_to_bf16.py - build the DSV4 Flash CAPPED BF16 master.

DISPOSABLE encode input, not a keeper (base + LoRA regenerate it in
under an hour): streams the abliterated FP8 master (E4M3 + E8M0 block
scales + I8-packed MXFP4 experts), dequants every weight to bf16, folds
the trained cap LoRA (W_eff = W + SCALE * B @ A, fp32 accumulate) at its
adapter sites, drops the now-meaningless .scale tensors, regenerates
model.safetensors.index.json, and writes FOLD_RECEIPT.json.

Dtype dispatch is explicit and fail-closed: any meta dtype without a
handler aborts the run; any LoRA site not found in the master aborts;
no tensor is ever written under a dtype it was not deliberately
converted to (the v1 draft leaked 712 BF16 natives as raw U16 and 35k
scales as U8 - caught by adversarial review before any real run).
"""

import argparse
import json
import os
import shutil
import time

import torch
from safetensors.torch import save_file

from full_loader import StreamingDSV4

SCALE = 16 / 8  # ALPHA / R from tc_lora_train (do not drift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="/wd/cap/lora.safetensors")
    ap.add_argument("--out", default="/workspace/model-capped-bf16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shards", default=None,
                    help="1-based inclusive range 'FIRST-LAST' of shards "
                         "to process (for multi-GPU striding; markers keep "
                         "concurrent workers disjoint)")
    args = ap.parse_args()

    from safetensors.torch import load_file
    lora_raw = load_file(args.lora)
    lora = {}
    for k, v in lora_raw.items():
        if not (k.startswith("lora.") and (k.endswith(".A") or k.endswith(".B"))):
            continue
        site = k[len("lora."):-2]
        d = lora.setdefault(site, {})
        d[k[-1]] = v.float()
    unpaired = sorted(k for k, v in lora.items()
                      if not ("A" in v and "B" in v))
    if unpaired:
        raise SystemExit(f"[fold] ABORT unpaired A/B keys: {unpaired[:5]}")
    if not lora:
        raise SystemExit("[fold] ABORT: no LoRA sites parsed")
    lora = {k: (v["A"], v["B"]) for k, v in lora.items()}
    print(f"[fold] {len(lora)} adapter sites, scale {SCALE}", flush=True)

    L = StreamingDSV4(device=args.device)
    os.makedirs(args.out, exist_ok=True)

    non_scale = {n for n, (dt, _, _) in L.meta.items() if dt != "F8_E8M0"}
    ghost = sorted(set(lora) - non_scale)
    if ghost:
        raise SystemExit(f"[fold] ABORT: LoRA sites absent from master "
                         f"(pre-check, no work burned): {ghost[:5]}")

    by_shard = {}
    for name, (dt, shape, shard) in L.meta.items():
        if dt == "F8_E8M0":
            continue  # block scales are meaningless once weights are dequant
        by_shard.setdefault(shard, []).append((name, dt))
    shards_sorted = sorted(by_shard, key=lambda s: int(
        "".join(c for c in s if c.isdigit()) or 0))
    if args.shards:
        lo, hi = (int(x) for x in args.shards.split("-"))
        shards_sorted = shards_sorted[lo - 1:hi]
        print(f"[fold] shard range {lo}-{hi} ({len(shards_sorted)} shards)",
              flush=True)

    receipts = {"scale": SCALE, "sites": {}, "dtype_census": {},
                "shards": len(shards_sorted)}
    weight_map = {}
    folded = set()
    census = receipts["dtype_census"]
    t0 = time.time()
    for si, shard in enumerate(shards_sorted):
        marker = os.path.join(args.out, shard + ".done")
        if os.path.exists(marker):
            for name, dt in by_shard[shard]:
                weight_map[name] = shard
                folded.add(name) if name in lora else None
                census[dt] = census.get(dt, 0) + 1
            print(f"[fold] shard {si+1}/{len(shards_sorted)} cached "
                  f"(receipts cover live shards only)", flush=True)
            continue
        out = {}
        for name, dt in by_shard[shard]:
            census[dt] = census.get(dt, 0) + 1
            if dt == "F8_E4M3":
                w = L.dequant_fp8(name).float()
                keep_dtype = torch.bfloat16
            elif dt == "I8":
                w = L.dequant_mxfp4(name).float()
                keep_dtype = torch.bfloat16
            elif dt == "BF16":
                w = L.passthrough(name).float()
                keep_dtype = torch.bfloat16
            elif dt == "F32":
                w = L.passthrough(name).float()
                keep_dtype = torch.float32  # natives keep their precision
            elif dt == "I64":
                raw, _, _ = L._raw(name)
                out[name] = raw
                weight_map[name] = shard
                continue
            else:
                raise SystemExit(f"[fold] unhandled dtype {dt} for {name}")
            if name in lora:
                A, B = lora[name]
                delta = (B.to(w.device) @ A.to(w.device)) * SCALE
                if delta.shape != w.shape:
                    raise SystemExit(
                        f"[fold] SHAPE MISMATCH {name}: W {tuple(w.shape)} "
                        f"vs delta {tuple(delta.shape)}")
                receipts["sites"][name] = {
                    "rel_fro": round(
                        (delta.norm() / w.norm()).item(), 6),
                    "abs_max_delta": round(
                        delta.abs().max().item(), 6),
                }
                w = w + delta
                folded.add(name)
            out[name] = w.to(keep_dtype)
            weight_map[name] = shard
            del w
        path = os.path.join(args.out, shard)
        save_file(out, path, metadata={"format": "pt"})
        open(marker, "w").write("ok")
        del out
        torch.cuda.empty_cache()
        print(f"[fold] shard {si+1}/{len(shards_sorted)} ({shard}) written, "
              f"{time.time()-t0:.0f}s", flush=True)

    if args.shards:
        print(f"[fold] range complete; finalizer skipped "
              f"(full-range run writes index/receipts)", flush=True)
        return

    missing = sorted(set(lora) - folded)
    if missing:
        raise SystemExit(
            f"[fold] ABORT: {len(missing)} LoRA sites not found in master: "
            f"{missing[:5]}")

    # configs/tokenizer: plain copies (never hardlink - shared inodes with
    # the source invite cross-artifact mutation)
    for f in os.listdir(L.dir):
        if f.endswith((".json", ".jinja", ".model", ".txt")) and \
                not f.startswith("model-"):
            src = os.path.join(L.dir, f)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(args.out, f))
    # config dtype must describe the actual master
    cfg_path = os.path.join(args.out, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["torch_dtype"] = "bfloat16"
    cfg.pop("quantization_config", None)
    json.dump(cfg, open(cfg_path, "w"), indent=1)

    # total_size must describe the WRITTEN tensors, not the source: FP8 and
    # packed-MXFP4 become bf16 (2 bytes/elem; I8 unpacks to 2x numel)
    def _out_bytes(name, dt):
        shape = L.meta[name][1]
        n = 1
        for d in shape:
            n *= d
        if dt == "F8_E4M3":
            return 2 * n
        if dt == "I8":
            return 2 * 2 * n  # packed nibbles -> bf16 elements
        if dt == "BF16":
            return 2 * n
        if dt == "F32":
            return 4 * n
        if dt == "I64":
            return 8 * n
        raise SystemExit(f"[fold] size-map unhandled dtype {dt} for {name}")
    total_size = sum(_out_bytes(n, L.meta[n][0]) for n in weight_map)
    index = {"metadata": {"total_size": total_size},
             "weight_map": dict(sorted(weight_map.items()))}
    json.dump(index, open(os.path.join(
        args.out, "model.safetensors.index.json"), "w"), indent=1)

    with open(os.path.join(args.out, "FOLD_RECEIPT.json"), "w") as f:
        json.dump(receipts, f, indent=1)
    rels = [v["rel_fro"] for v in receipts["sites"].values()]
    print(f"[fold] DONE {len(shards_sorted)} shards, {len(folded)}/{len(lora)} "
          f"sites folded, {sum(census.values())} tensors, "
          f"{time.time()-t0:.0f}s", flush=True)
    if rels:
        rt = torch.tensor(rels)
        print(f"[fold] rel_fro: min {rt.min():.5f} med {rt.median():.5f} "
              f"max {rt.max():.5f} | census {census}", flush=True)


if __name__ == "__main__":
    main()
