#!/usr/bin/env python3
"""glm_peek.py — build tensor_meta.json for the GLM-5.3-Flash FP8 checkpoint.

Scans every safetensors shard header in the model dir and emits the same
{meta, offsets} format as the DSV4 harness (full_loader.py):
  meta:    name -> [dtype_str, shape, shard_file]
  offsets: name -> [shard_file, abs_data_start, nbytes]
visual.* and model.language_model.layers.45.* (the MTP/nextn layer) are
SKIPPED: the ThinkingCap engine is text-only and freezes the drafter
(drafter-anchor lesson). Runs anywhere (stdlib only)."""

import json
import os
import struct
import sys

MODEL_DIR = os.environ.get("GLM_MODEL", "/root/glm-fp8")
OUT = os.environ.get("GLM_META", "/root/tc-glm/tensor_meta.json")

SKIP_PREFIXES = ("model.visual.", "model.language_model.layers.45.")


def shard_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    data_start = 8 + n
    return hdr, data_start


def main():
    meta, offsets = {}, {}
    shards = sorted(f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors"))
    for sh in shards:
        hdr, base = shard_header(os.path.join(MODEL_DIR, sh))
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            if any(name.startswith(p) for p in SKIP_PREFIXES):
                continue
            s, e = info["data_offsets"]
            meta[name] = [info["dtype"], info["shape"], sh]
            offsets[name] = [sh, base + s, e - s]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"meta": meta, "offsets": offsets}, open(OUT, "w"))
    n_f8 = sum(1 for v in meta.values() if v[0] == "F8_E4M3")
    print(f"[peek] {len(meta)} tensors ({n_f8} FP8) from {len(shards)} shards -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
