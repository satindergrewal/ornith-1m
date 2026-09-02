#!/usr/bin/env python3
"""ThinkingCap dummy cycle - STEP 01: synthesize the nano-DSV4-Vision checkpoint.

Reads the REAL DeepSeek-V4-Flash-Vision-Exp tensor index (72,633 names),
maps each name to the nano architecture (4 layers, 8 experts, vision 2
blocks, 3 nextn kept), derives shapes from nano-config.json math, and
writes a loadable BF16 HF checkpoint: nano-dsv4-vision/.

Determinism: torch.manual_seed(20260902). Coverage is logged: any real
name pattern that cannot be mapped is counted and reported - the goal is
100% coverage of kept names.

Run (box, no GPU needed):
  docker run --rm -v /mnt/t5evo/thinkingcap-dummy:/wd -w /wd \
    --entrypoint python3 glm53-k35-tp2capture 01_synth.py --execute
"""
import argparse, json, os, re, sys
from pathlib import Path
import torch

WD = Path(os.environ.get("SYNTH_WD", "/wd"))
NANO_CFG = json.loads((WD / "nano-config.json").read_text())

H = NANO_CFG["hidden_size"]            # 256
HEADS = NANO_CFG["num_attention_heads"]  # 8
HD = NANO_CFG["head_dim"]              # 64
QKR = NANO_CFG["qk_rope_head_dim"]     # 16
QLR = NANO_CFG["q_lora_rank"]          # 64
OLR = NANO_CFG["o_lora_rank"]          # 64
OG = NANO_CFG["o_groups"]              # 2
KVLR = 64                              # shrunk kv_lora_rank (real 512)
IN_H = NANO_CFG["index_n_heads"]       # 4
IN_D = NANO_CFG["index_head_dim"]      # 32
IN_K = NANO_CFG["index_topk"]          # 16
MOE_I = NANO_CFG["moe_intermediate_size"]  # 64
N_EXP = NANO_CFG["n_routed_experts"]   # 8
N_LAYERS = NANO_CFG["num_hidden_layers"]  # 4
V_DIM = NANO_CFG["vision_dim"]         # 64
V_I = NANO_CFG["vision_inter_dim"]     # 128
V_H = NANO_CFG["vision_n_heads"]       # 4
V_L = NANO_CFG["vision_n_layers"]      # 2
VOCAB = NANO_CFG["vocab_size"]         # 129280
DRA = NANO_CFG.get("vision_downsample_ratio", 3)
MARKOV = 64                            # shrunk dspark_markov_rank (real 256)
L0 = 10                                # nano layer k <- real layer L0+k; same
                                       # window as 01b (ablit arm) so both A/B
                                       # arms share one architecture

def nano_name_and_shape(real_name, layer_idx=None):
    """Map one real tensor name to (nano_name, shape) or None to skip."""
    n = real_name
    m = re.match(r"^layers\.(\d+)\.(.*)$", n)
    if m:
        i, rest = int(m.group(1)), m.group(2)
        if i < L0 or i >= L0 + N_LAYERS:
            return None
        s = _layer_shape(rest, i)
        return (f"layers.{i - L0}.{rest}", s) if s is not None else None
    m = re.match(r"^vision\.blocks\.(\d+)\.(.*)$", n)
    if m:
        i, rest = int(m.group(1)), m.group(2)
        if i >= V_L:
            return None
        s = _vision_shape(rest)
        return (f"vision.blocks.{i}.{rest}", s) if s is not None else None
    if n.startswith("mtp."):
        s = _mtp_shape(n[4:])
        return (n, s) if s is not None else None
    return (n, _top_shape(n)) if n in _TOP else None

_LAYER_SHAPES = {
    "attn.wq_a.weight": (QLR, H),
    "attn.wq_a.scale": (),
    "attn.wq_b.weight": (HEADS * (HD + QKR), QLR),
    "attn.wq_b.scale": (),
    "attn.wkv.weight": (KVLR + QKR, H),
    "attn.wkv.scale": (),
    "attn.wo_a.weight": (OLR, HEADS * HD // OG),
    "attn.wo_a.scale": (),
    "attn.wo_b.weight": (H, OLR),
    "attn.wo_b.scale": (),
    "attn.q_norm.weight": (QLR,),
    "attn.kv_norm.weight": (KVLR,),
    "attn.attn_sink": (HEADS,),
    "attn_norm.weight": (H,),
    "ffn_norm.weight": (H,),
    "ffn.gate.weight": (N_EXP, H),
    "ffn.gate.bias": (N_EXP,),
    "ffn.gate.bias_vl": (N_EXP,),
    "hc_attn_base": (4,), "hc_attn_fn": (4,), "hc_attn_scale": (4,),
    "hc_ffn_base": (4,), "hc_ffn_fn": (4,), "hc_ffn_scale": (4,),
    "ffn.shared_experts.w1.weight": (MOE_I, H),
    "ffn.shared_experts.w1.scale": (),
    "ffn.shared_experts.w2.weight": (H, MOE_I),
    "ffn.shared_experts.w2.scale": (),
    "ffn.shared_experts.w3.weight": (MOE_I, H),
    "ffn.shared_experts.w3.scale": (),
    "attn.indexer.wq_b.weight": (IN_H * IN_D, QLR),
    "attn.indexer.wq_b.scale": (),
    "attn.indexer.weights_proj.weight": (IN_K, IN_H),
    "attn.indexer.compressor.wkv.weight": (KVLR + QKR, H),
    "attn.indexer.compressor.wgate.weight": (1, KVLR),
    "attn.indexer.compressor.norm.weight": (KVLR,),
    "attn.indexer.compressor.ape": (1, 128, QKR),
    "attn.compressor.wkv.weight": (KVLR + QKR, H),
    "attn.compressor.wgate.weight": (1, KVLR),
    "attn.compressor.norm.weight": (KVLR,),
    "attn.compressor.ape": (1, 128, QKR),
}

def _layer_shape(rest, i):
    if rest in _LAYER_SHAPES:
        return _LAYER_SHAPES[rest]
    m2 = re.match(r"^ffn\.experts\.(\d+)\.(w\d)\.(weight|scale)$", rest)
    if m2:
        e, w, kind = int(m2.group(1)), m2.group(2), m2.group(3)
        if e >= N_EXP:
            return None
        if kind == "scale":
            return ()
        return {"w1": (MOE_I, H), "w2": (H, MOE_I), "w3": (MOE_I, H)}[w]
    if rest.startswith("ffn.gate.tid"):  # tid2eid-style router tables, few layers only
        return (N_EXP,)
    return None

_VISION_SHAPES = {
    "attn.wqkv.weight": (3 * V_DIM, V_DIM),
    "attn.wqkv.bias": (3 * V_DIM,),
    "attn.wo.weight": (V_DIM, V_DIM),
    "attn.wo.bias": (V_DIM,),
    "mlp.w1.weight": (V_I, V_DIM),
    "mlp.w2.weight": (V_DIM, V_I),
    "norm1.weight": (V_DIM,),
    "norm2.weight": (V_DIM,),
}

def _vision_shape(rest):
    return _VISION_SHAPES.get(rest)

_MTP_SHAPES = {
    "main_norm.weight": (H,),
    "main_proj.weight": (2 * H, H),
    "main_proj.scale": (),
    "norm.weight": (H,),
    "ffn_norm.weight": (H,),
    "attn.wq_a.weight": (QLR, H),
    "attn.wq_a.scale": (),
    "attn.wq_b.weight": (HEADS * (HD + QKR), QLR),
    "attn.wq_b.scale": (),
    "attn.wkv.weight": (KVLR + QKR, H),
    "attn.wkv.scale": (),
    "attn.wo_a.weight": (OLR, HEADS * HD // OG),
    "attn.wo_a.scale": (),
    "attn.wo_b.weight": (H, OLR),
    "attn.wo_b.scale": (),
    "attn.q_norm.weight": (QLR,),
    "attn.kv_norm.weight": (KVLR,),
    "attn.attn_sink": (HEADS,),
    "attn_norm.weight": (H,),
    "ffn.gate.weight": (N_EXP, H),
    "ffn.gate.bias": (N_EXP,),
    "ffn.gate.bias_vl": (N_EXP,),
    "ffn.shared_experts.w1.weight": (MOE_I, H),
    "ffn.shared_experts.w1.scale": (),
    "ffn.shared_experts.w2.weight": (H, MOE_I),
    "ffn.shared_experts.w2.scale": (),
    "ffn.shared_experts.w3.weight": (MOE_I, H),
    "ffn.shared_experts.w3.scale": (),
    "hc_attn_base": (4,), "hc_attn_fn": (4,), "hc_attn_scale": (4,),
    "hc_ffn_base": (4,), "hc_ffn_fn": (4,), "hc_ffn_scale": (4,),
    "confidence_head.proj.weight": (H, H),
    "markov_head.markov_w1.weight": (MARKOV, H + VOCAB if False else H),
    "markov_head.markov_w2.weight": (VOCAB, MARKOV),
    "hc_head_base": (4,), "hc_head_fn": (4,), "hc_head_scale": (4,),
    "hc_attn_base": (4,), "hc_attn_fn": (4,), "hc_attn_scale": (4,),
    "hc_ffn_base": (4,), "hc_ffn_fn": (4,), "hc_ffn_scale": (4,),
}

def _mtp_shape(rest):
    if rest in _MTP_SHAPES:
        return _MTP_SHAPES[rest]
    # mtp.{layer}.<rest> -> strip the layer index (already in nano name)
    m2 = re.match(r"^(\d+)\.(.*)$", rest)
    if m2:
        inner = m2.group(2)
        if inner in _MTP_SHAPES:
            return _MTP_SHAPES[inner]
        m3 = re.match(r"^ffn\.experts\.(\d+)\.(w\d)\.(weight|scale)$", inner)
        if m3:
            e, w, kind = int(m3.group(1)), m3.group(2), m3.group(3)
            if e >= N_EXP:
                return None
            if kind == "scale":
                return ()
            return {"w1": (MOE_I, H), "w2": (H, MOE_I), "w3": (MOE_I, H)}[w]
    return None

_TOP = {
    "embed.weight", "head.weight", "norm.weight",
    "hc_head_base", "hc_head_fn", "hc_head_scale",
    "image_pad", "image_start", "image_end", "image_newline",
    "aligner.w1.weight", "aligner.w1.bias", "aligner.w2.weight", "aligner.w2.bias",
    "vision.norm.weight", "vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias",
}

def _top_shape(n):
    return {
        "embed.weight": (VOCAB, H), "head.weight": (VOCAB, H), "norm.weight": (H,),
        "hc_head_base": (4,), "hc_head_fn": (4,), "hc_head_scale": (4,),
        "image_pad": (H,), "image_start": (H,), "image_end": (H,), "image_newline": (H,),
        "aligner.w1.weight": (H, DRA * DRA * V_DIM), "aligner.w1.bias": (H,),
        "aligner.w2.weight": (H, H), "aligner.w2.bias": (H,),
        "vision.norm.weight": (V_DIM,),
        "vision.patch_embed.proj.weight": (V_DIM, 3 * 14 * 14),
        "vision.patch_embed.proj.bias": (V_DIM,),
    }.get(n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    idx = json.loads((WD / "vision-exp-index.json").read_text())
    real_names = sorted(idx["weight_map"].keys())
    mapped, skipped_patterns = {}, {}
    real_prefix, mapped_prefix = {}, {}
    for rn in real_names:
        p = rn.split(".")[0]
        real_prefix[p] = real_prefix.get(p, 0) + 1
        got = nano_name_and_shape(rn)
        if got is None:
            pat = re.sub(r"\d+", "N", rn)
            skipped_patterns[pat] = skipped_patterns.get(pat, 0) + 1
            continue
        mapped[got[0]] = got[1]
        mapped_prefix[p] = mapped_prefix.get(p, 0) + 1
    total_params = sum(torch.tensor(s).prod().item() if s else 1 for s in mapped.values())
    report = {
        "step": "01_synth",
        "real_tensors": len(real_names),
        "nano_tensors": len(mapped),
        "real_by_prefix": real_prefix,
        "mapped_by_prefix": mapped_prefix,
        "skipped_patterns": skipped_patterns,
        "est_params_M": round(total_params / 1e6, 2),
        "dry_run": not a.execute,
    }
    print(json.dumps(report, indent=1)[:8000])
    if not a.execute:
        return 0
    torch.manual_seed(20260902)
    out = WD / "nano-dsv4-vision"
    out.mkdir(exist_ok=True)
    tensors = {}
    for name, shape in sorted(mapped.items()):
        if shape == ():
            tensors[name] = torch.ones((), dtype=torch.float32)  # scales = 1.0
        elif name.endswith(".bias") or "ape" in name:
            tensors[name] = torch.zeros(shape, dtype=torch.bfloat16)
        elif name.endswith("norm.weight") or name.endswith(("_norm.weight",)) or name in ("norm.weight",):
            tensors[name] = torch.ones(shape, dtype=torch.bfloat16)
        else:
            tensors[name] = (torch.randn(shape) * NANO_CFG["initializer_range"]).to(torch.bfloat16)
    from safetensors.torch import save_file
    shard = 0
    names = sorted(tensors)
    weight_map, per = {}, 800
    for s in range(0, len(names), per):
        shard += 1
        fname = f"model-{shard:05d}-of-{(len(names)+per-1)//per:05d}.safetensors"
        chunk = {n: tensors[n] for n in names[s:s+per]}
        save_file(chunk, str(out / fname))
        for n in chunk:
            weight_map[n] = fname
    index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
             "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    (out / "config.json").write_text(json.dumps(NANO_CFG, indent=1))
    print(json.dumps({"written": str(out), "shards": shard, "tensors": len(tensors),
                      "est_params_M": report["est_params_M"]}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
