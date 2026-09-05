#!/usr/bin/env python3
"""analyze_meta.py - full structural analysis of tensor_meta.json."""
import json, re, collections

d = json.load(open("/wd/tensor_meta.json"))
meta = d["meta"]

cfg = json.load(open("/model/config.json"))
cr = cfg["compress_ratios"]
print("compress_ratios len:", len(cr), "per-layer 0..42:", cr[:43], "tail:", cr[43:])
print("n_layers:", cfg["num_hidden_layers"], "experts:", cfg["n_routed_experts"],
      "topk:", cfg["num_experts_per_tok"])

def names_for(prefix):
    return {k: meta[k] for k in meta if k.startswith(prefix)}

# Full attn listing for layer types: 0 (sliding?), 2 (csa), 3 (hca), 42 (last)
for L in [0, 1, 2, 3, 42]:
    ns = names_for(f"layers.{L}.")
    attn = {k: v for k, v in ns.items() if ".attn." in k or k.endswith("attn_sink")}
    print(f"--- layers.{L} attn tensors ({len(attn)}):")
    for k in sorted(attn):
        print("   ", k, attn[k][0], attn[k][1])

# hc + ffn top for layer 10
for pat in ["layers.10.hc_", "layers.10.ffn.gate", "layers.10.ffn.shared_experts",
            "layers.2.hc_", "layers.3.hc_"]:
    ns = {k: v for k, v in meta.items() if k.startswith(pat)}
    print(f"--- {pat} ({len(ns)}):")
    for k in sorted(ns)[:14]:
        print("   ", k, ns[k][0], ns[k][1])

# top-level
print("--- top-level:", {k: (meta[k][0], meta[k][1]) for k in meta
                         if not k.startswith(("layers.", "mtp.", "vision."))})

# mtp.0 structure: distinct suffix patterns
mtp0 = names_for("mtp.0.")
pats = collections.Counter(re.sub(r"\d+", "N", k) for k in mtp0)
print("--- mtp.0 patterns:", len(pats))
for p, c in sorted(pats.items()):
    if "experts" not in p or c <= 8:
        print(f"    {c:4d}  {p}  {meta.get(p.replace('mtp.0.', 'mtp.0.'), '')}")
print("--- mtp.0 non-attn/ffn/hc extras:")
for k in sorted(mtp0):
    if not re.search(r"\.(attn|ffn|hc_attn|hc_ffn)\.", k) and "experts" not in k:
        print("   ", k, mtp0[k][0], mtp0[k][1])

# expert scale/weight shape relation + dtype check across experts
for k in ["layers.10.ffn.experts.0.w1.weight", "layers.10.ffn.experts.0.w1.scale",
          "layers.10.ffn.experts.0.w2.weight", "layers.10.ffn.experts.0.w2.scale",
          "layers.10.ffn.experts.0.w3.weight", "layers.10.ffn.experts.0.w3.scale",
          "layers.10.ffn.shared_experts.w1.weight", "layers.10.ffn.shared_experts.w1.scale",
          "layers.10.ffn.gate.weight", "layers.10.ffn.gate.bias",
          "layers.10.attn.wq_b.weight", "layers.10.attn.wq_b.scale",
          "layers.10.attn.compressor.norm.weight", "layers.10.attn.compressor.ape",
          "layers.11.attn.compressor.wkv.weight", "layers.11.attn.compressor.wgate.weight",
          "layers.11.attn.compressor.ape", "layers.11.attn.compressor.norm.weight",
          "layers.2.attn.compressor.ape", "layers.2.attn.indexer.compressor.ape",
          "layers.2.attn.indexer.compressor.norm.weight",
          "layers.10.hc_attn_base", "layers.10.hc_attn_scale", "layers.10.hc_ffn_base",
          "layers.10.hc_ffn_scale", "hc_head_base", "hc_head_scale"]:
    print("   ", k, "->", meta.get(k))

# per-layer-group byte sizes (from offsets)
offs = d["offsets"]
def bytes_of(pred):
    tot = sum(v[2] for k, v in offs.items() if pred(k))
    return tot
lang = bytes_of(lambda k: k.startswith("layers."))
per_exp = bytes_of(lambda k: re.match(r"layers\.10\.ffn\.experts\.", k))
per_l10 = bytes_of(lambda k: k.startswith("layers.10."))
print(f"bytes: lang layers total {lang/1e9:.2f} GB; layer10 total {per_l10/1e9:.3f} GB; "
      f"layer10 experts {per_exp/1e9:.3f} GB; layer10 non-expert {(per_l10-per_exp)/1e6:.1f} MB")
print(f"bytes: embed {bytes_of(lambda k: k=='embed.weight')/1e9:.3f} GB; "
      f"head {bytes_of(lambda k: k=='head.weight')/1e9:.3f} GB; "
      f"mtp {bytes_of(lambda k: k.startswith('mtp.'))/1e9:.2f} GB; "
      f"vision {bytes_of(lambda k: k.startswith('vision.'))/1e9:.2f} GB")
# how many shards does one layer span
shards_l10 = sorted({offs[k][0] for k in offs if k.startswith("layers.10.")})
print("layer10 shards:", len(shards_l10), shards_l10)
