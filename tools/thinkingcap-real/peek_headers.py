#!/usr/bin/env python3
"""peek_headers.py - read safetensors headers of the real checkpoint.
No torch needed: header = u64 LE length + JSON. Emits name -> [dtype, shape]
for a filtered prefix set, plus shard->header byte offsets (for later mmap
streaming without re-parsing)."""
import json, os, struct, sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/model"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/wd/tensor_meta.json"

def shard_headers(d):
    meta = {}
    offsets = {}   # name -> (shard, data_start_abs, nbytes, dtype, shape)
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".safetensors"):
            continue
        p = os.path.join(d, fn)
        with open(p, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            s, e = info["data_offsets"]
            meta[name] = [info["dtype"], info["shape"], fn]
            offsets[name] = [fn, base + s, e - s]
    return meta, offsets

meta, offs = shard_headers(MODEL)
json.dump({"meta": meta, "offsets": offs}, open(OUT, "w"))
print("tensors:", len(meta))

# report the interesting shapes compactly
def show(pat, n=6):
    hits = {k: v for k, v in meta.items() if pat in k}
    print(f"--- {pat} ({len(hits)} hits)")
    for k in sorted(hits)[:n]:
        print("   ", k, hits[k])

import re
# one CSA layer (10, even -> indexer), one HCA (11)
for pat in ["layers.10.attn", "layers.11.attn", "layers.10.hc", "layers.10.ffn.gate",
            "layers.10.ffn.shared", "layers.10.ffn.experts.0.", "layers.10.ffn.experts.1.",
            "hc_head", "embed.weight", "head.weight", "norm.weight",
            "layers.0.attn", "mtp.0.", "mtp.2."]:
    show(pat, 8)

# dtypes census
dts = {}
for v in meta.values():
    dts[v[0]] = dts.get(v[0], 0) + 1
print("dtypes:", dts)
# expert count + per-expert shapes
exp = [k for k in meta if re.match(r"layers\.10\.ffn\.experts\.(\d+)\.", k)]
ids = sorted({int(re.match(r"layers\.10\.ffn\.experts\.(\d+)\.", k).group(1)) for k in exp})
print("layer10 experts:", len(ids), "min/max", ids[0], ids[-1])
# which layers have indexer
idx_layers = sorted({int(m.group(1)) for k in meta if (m := re.match(r"layers\.(\d+)\.attn\.indexer\.", k))})
print("indexer layers:", idx_layers)
# mtp layer names sample
mtp = sorted(k for k in meta if k.startswith("mtp."))
print("mtp tensor count:", len(mtp))
print("mtp sample:", mtp[:10])
# scale naming for one expert tensor + wo_b
for k in ["layers.10.ffn.experts.0.w1.weight", "layers.10.attn.wo_b.weight",
          "layers.10.attn.wq_a.weight", "layers.10.attn.compressor.wkv.weight",
          "layers.10.attn.compressor.wgate.weight", "layers.10.attn.compressor.ape",
          "layers.10.attn.indexer.compressor.wgate.weight",
          "layers.10.attn.indexer.weights_proj.weight", "layers.10.attn.indexer.wq_b.weight",
          "layers.10.hc_attn_fn", "hc_head_fn"]:
    print(k, "->", meta.get(k))
