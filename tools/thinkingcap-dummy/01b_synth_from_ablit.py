#!/usr/bin/env python3
"""ThinkingCap dummy cycle - STEP 01b: nano dummy FROM the ablit base.

Slices the nano-DSV4-Vision dummy OUT OF the drowzeys community-abliterated
DeepSeek-V4-Flash-Vision-Exp checkpoint (/mnt/t5evo/dsv4-vision-ablit,
FP8 block-scale), so the dummy starts life already-abliterated and
rehearses the real cap-on-ablit pipeline.

Layer remap: nano layer k <- real layer 10+k (the drowzeys wo_b edit
covers L10-35, so all four nano layers carry the actual edit).
FP8 tensors are dequantized (block-wise 128, per-row, or scalar scale)
then sliced to nano dims; the result is a BF16 checkpoint with the SAME
607-name tensor set 01_synth.py produces.

Deterministic slicing (leading rows/cols). No GPU needed.

Run (box):
  docker run --rm -v /mnt/t5evo/thinkingcap-dummy:/wd -v /mnt/t5evo/dsv4-vision-ablit:/src:ro \
    -w /wd --entrypoint python3 glm53-k35-tp2capture 01b_synth_from_ablit.py --execute
"""
import argparse
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WD = Path(os.environ.get("SYNTH_WD", "/wd"))
SRC = Path(os.environ.get("ABLIT_SRC", "/src"))
NANO_CFG = json.loads((WD / "nano-config.json").read_text())

H = NANO_CFG["hidden_size"]            # 256
HEADS = NANO_CFG["num_attention_heads"]  # 8
HD = NANO_CFG["head_dim"]              # 64
QKR = NANO_CFG["qk_rope_head_dim"]     # 16
QLR = NANO_CFG["q_lora_rank"]          # 64
OLR = NANO_CFG["o_lora_rank"]          # 64
OG = NANO_CFG["o_groups"]              # 2
KVLR = 64
IN_H = NANO_CFG["index_n_heads"]       # 4
IN_D = NANO_CFG["index_head_dim"]      # 32
IN_K = NANO_CFG["index_topk"]          # 16
MOE_I = NANO_CFG["moe_intermediate_size"]  # 64
N_EXP = NANO_CFG["n_routed_experts"]   # 8
N_LAYERS = NANO_CFG["num_hidden_layers"]  # 4
V_DIM = NANO_CFG["vision_dim"]         # 64
V_I = NANO_CFG["vision_inter_dim"]     # 128
V_L = NANO_CFG["vision_n_layers"]      # 2
VOCAB = NANO_CFG["vocab_size"]
MARKOV = 64
L0 = 10  # nano layer k <- real layer L0+k (drowzeys edited L10-35)

# real dims (DeepSeek-V4-Flash-Vision-Exp)
R_H = 4096
R_HEADS = 64
R_HD = 512
R_QKR = 64
R_QLR = 1024
R_OLR = 1024
R_OG = 8
R_KVLR = 512
R_IN_H = 64
R_IN_D = 128
R_IN_K = 512
R_MOE_I = 2048
R_V_DIM = 1024
R_V_I = 2816


def slice2(t, n0, n1):
    """Slice a 2-D weight (rows, cols) to nano dims."""
    if t.ndim == 2:
        return t[:n0, :n1].contiguous()
    return t


def dequant_if_fp8(t, scale):
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if scale is None:
            return t.to(torch.bfloat16)
        s = scale.float()
        tq = t.float()
        if s.ndim == 0:
            return (tq * s).to(torch.bfloat16)
        if s.ndim == 1:
            return (tq * s.view(-1, 1)).to(torch.bfloat16)
        # block-wise 128 over the last two dims
        bs = 128
        n, k = tq.shape[-2], tq.shape[-1]
        sb = s.reshape((-1,) + s.shape[-2:]) if s.ndim > 2 else s
        sb = sb[:, : (n + bs - 1) // bs, : (k + bs - 1) // bs] if sb.ndim == 3 else sb
        rep = sb.repeat_interleave(bs, -2).repeat_interleave(bs, -1)
        rep = rep[..., :n, :k]
        return (tq * rep).to(torch.bfloat16)
    return t.to(torch.bfloat16)


def nano_for_real(name):
    """(real_name) -> (nano_name, slicer) or None to skip."""
    n = name
    m = re.match(r"^model\.language_model\.layers\.(\d+)\.(.*)$", n) or \
        re.match(r"^layers\.(\d+)\.(.*)$", n)
    if m:
        i, rest = int(m.group(1)), m.group(2)
        if i < L0 or i >= L0 + N_LAYERS:
            return None
        return (f"layers.{i - L0}.{rest}", rest)
    m = re.match(r"^model\.visual\.blocks\.(\d+)\.(.*)$", n) or \
        re.match(r"^vision\.blocks\.(\d+)\.(.*)$", n)
    if m:
        i, rest = int(m.group(1)), m.group(2)
        if i >= V_L:
            return None
        return (f"vision.blocks.{i}.{rest}", "vision." + rest)
    if n.startswith("mtp.") or n.startswith("model.language_model.mtp."):
        rest = n.split("mtp.", 1)[1]
        return ("mtp." + rest, "mtp." + rest)
    top = _TOP_MAP.get(n)
    if top:
        return top
    return None


_TOP_MAP = {
    "hc_head_base": ("hc_head_base", "hc4"),
    "hc_head_fn": ("hc_head_fn", "hc4"),
    "hc_head_scale": ("hc_head_scale", "hc4"),
    "model.language_model.embed_tokens.weight": ("embed.weight", "embed"),
    "embed.weight": ("embed.weight", "embed"),
    "model.lm_head.weight": ("head.weight", "embed"),
    "head.weight": ("head.weight", "embed"),
    "model.language_model.norm.weight": ("norm.weight", "h"),
    "norm.weight": ("norm.weight", "h"),
    "image_pad": ("image_pad", "h"),
    "image_start": ("image_start", "h"),
    "image_end": ("image_end", "h"),
    "image_newline": ("image_newline", "h"),
    "model.visual.aligner.w1.weight": ("aligner.w1.weight", "aligner.w1w"),
    "aligner.w1.weight": ("aligner.w1.weight", "aligner.w1w"),
    "model.visual.aligner.w1.bias": ("aligner.w1.bias", "aligner.w1b"),
    "aligner.w1.bias": ("aligner.w1.bias", "aligner.w1b"),
    "model.visual.aligner.w2.weight": ("aligner.w2.weight", "aligner.w2w"),
    "aligner.w2.weight": ("aligner.w2.weight", "aligner.w2w"),
    "model.visual.aligner.w2.bias": ("aligner.w2.bias", "aligner.w2b"),
    "aligner.w2.bias": ("aligner.w2.bias", "aligner.w2b"),
    "model.visual.norm.weight": ("vision.norm.weight", "v"),
    "vision.norm.weight": ("vision.norm.weight", "v"),
    "model.visual.patch_embed.proj.weight":
        ("vision.patch_embed.proj.weight", "patchw"),
    "vision.patch_embed.proj.weight":
        ("vision.patch_embed.proj.weight", "patchw"),
    "model.visual.patch_embed.proj.bias":
        ("vision.patch_embed.proj.bias", "patchb"),
    "vision.patch_embed.proj.bias":
        ("vision.patch_embed.proj.bias", "patchb"),
}


def apply_slice(kind, rest, t):
    """Slice dequantized BF16 tensor `t` by its role."""
    r = rest
    if r.endswith(".scale"):
        return None  # consumed by dequant; nano scales are ones
    if r == "norm.weight":
        return t[:H]
    if r == "confidence_head.proj.weight":
        return slice2(t, H, H)
    # ---- language layer roles ----
    if r == "self_attn.wq_a.weight" or r == "attn.wq_a.weight":
        return slice2(t, QLR, H)
    if r in ("self_attn.wq_b.weight", "attn.wq_b.weight"):
        return slice2(t, HEADS * (HD + QKR), QLR)
    if r in ("self_attn.wkv.weight", "attn.wkv.weight"):
        return slice2(t, KVLR + QKR, H)
    if r in ("self_attn.wo_a.weight", "attn.wo_a.weight"):
        return slice2(t, OLR, HEADS * HD // OG)
    if r in ("self_attn.wo_b.weight", "attn.wo_b.weight"):
        return slice2(t, H, OLR)
    if r in ("self_attn.q_norm.weight", "attn.q_norm.weight"):
        return t[:QLR]
    if r in ("self_attn.kv_norm.weight", "attn.kv_norm.weight"):
        return t[:KVLR]
    if r in ("self_attn.attn_sink", "attn.attn_sink"):
        return t[:HEADS]
    if r in ("input_layernorm.weight", "attn_norm.weight"):
        return t[:H]
    if r in ("post_attention_layernorm.weight", "ffn_norm.weight"):
        return t[:H]
    if r in ("mlp.gate.weight", "ffn.gate.weight"):
        return slice2(t, N_EXP, H)
    if r in ("mlp.gate.bias", "ffn.gate.bias"):
        return t[:N_EXP]
    if r in ("mlp.gate.bias_vl", "ffn.gate.bias_vl"):
        return t[:N_EXP]
    if r.startswith("mlp.gate.tid") or r.startswith("ffn.gate.tid"):
        return t[:N_EXP] if t.ndim == 1 else slice2(t, N_EXP, t.shape[-1])
    m2 = re.match(r"(?:mlp|ffn)\.shared_experts\.(w\d)\.(weight)", r)
    if m2:
        return {"w1": slice2(t, MOE_I, H), "w2": slice2(t, H, MOE_I),
                "w3": slice2(t, MOE_I, H)}[m2.group(1)]
    m2 = re.match(r"(?:mlp|ffn)\.experts\.(\d+)\.(w\d)\.(weight)", r)
    if m2:
        e, w = int(m2.group(1)), m2.group(2)
        if e >= N_EXP:
            return None
        return {"w1": slice2(t, MOE_I, H), "w2": slice2(t, H, MOE_I),
                "w3": slice2(t, MOE_I, H)}[w]
    if r in ("self_attn.indexer.wq_b.weight", "attn.indexer.wq_b.weight"):
        return slice2(t, IN_H * IN_D, QLR)
    if r in ("self_attn.indexer.weights_proj.weight",
             "attn.indexer.weights_proj.weight"):
        return slice2(t, IN_K, IN_H)
    if r in ("self_attn.indexer.compressor.wkv.weight",
             "attn.indexer.compressor.wkv.weight",
             "self_attn.compressor.wkv.weight",
             "attn.compressor.wkv.weight"):
        return slice2(t, KVLR + QKR, H)
    if r in ("self_attn.indexer.compressor.wgate.weight",
             "attn.indexer.compressor.wgate.weight",
             "self_attn.compressor.wgate.weight",
             "attn.compressor.wgate.weight"):
        return slice2(t, 1, KVLR)
    if r in ("self_attn.indexer.compressor.norm.weight",
             "attn.indexer.compressor.norm.weight",
             "self_attn.compressor.norm.weight",
             "attn.compressor.norm.weight"):
        return t[:KVLR]
    if r in ("self_attn.indexer.compressor.ape",
             "attn.indexer.compressor.ape",
             "self_attn.compressor.ape", "attn.compressor.ape"):
        if t.ndim == 3:
            return t[:, :, :QKR].contiguous()
        return t[:, :QKR].unsqueeze(0).contiguous()
    if r in ("hc_attn_base", "hc_attn_fn", "hc_attn_scale",
             "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale"):
        return t
    # ---- mtp roles ----
    if r.startswith("mtp."):
        inner = re.sub(r"^\d+\.", "", r[len("mtp."):] if r.startswith("mtp.") else r)
        if "main_norm" in r:
            return t[:H]
        if "main_proj" in r and r.endswith("weight"):
            return slice2(t, 2 * H, H)
        if "markov_w1" in r:
            return slice2(t, MARKOV, H)
        if "markov_w2" in r:
            return slice2(t, VOCAB, MARKOV)
        if r.endswith("hc_head_base") or r.endswith("hc_head_fn") or \
                r.endswith("hc_head_scale"):
            return t
        return apply_slice(kind, inner.replace("self_attn.", "attn.")
                           .replace("mlp.", "ffn."), t)
    # ---- vision roles ----
    if kind.startswith("vision."):
        vr = kind[len("vision."):]
        if vr == "attn.wqkv.weight" or vr == "wqkv.weight":
            return slice2(t, 3 * V_DIM, V_DIM)
        if vr in ("attn.wqkv.bias", "wqkv.bias"):
            return t[:3 * V_DIM]
        if vr in ("attn.wo.weight", "wo.weight"):
            return slice2(t, V_DIM, V_DIM)
        if vr in ("attn.wo.bias", "wo.bias"):
            return t[:V_DIM]
        if vr in ("mlp.w1.weight", "w1.weight"):
            return slice2(t, V_I, V_DIM)
        if vr in ("mlp.w2.weight", "w2.weight"):
            return slice2(t, V_DIM, V_I)
        if vr in ("norm1.weight", "norm2.weight"):
            return t[:V_DIM]
    if kind == "embed" or kind == "head":
        return slice2(t, VOCAB, H)
    if kind == "hc4":
        return t
    if kind == "h" or kind == "v":
        return t[:H] if kind == "h" else t[:V_DIM]
    if kind == "patchw":
        return slice2(t, V_DIM, t.shape[-1])
    if kind == "patchb":
        return t[:V_DIM]
    if kind == "aligner.w1w":
        return slice2(t, H, 9 * V_DIM)
    if kind == "aligner.w1b":
        return t[:H]
    if kind == "aligner.w2w":
        return slice2(t, H, H)
    if kind == "aligner.w2b":
        return t[:H]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()

    idx = json.loads((SRC / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    shard_of = {}
    for name, sh in wmap.items():
        shard_of.setdefault(sh, []).append(name)

    tensors, skipped = {}, {}
    for shard in sorted(shard_of):
        with safe_open(str(SRC / shard), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for rn in sorted(shard_of[shard]):
                got = nano_for_real(rn)
                if got is None:
                    continue
                nn, role = got
                if rn.endswith(".scale"):
                    m3 = re.match(r".*\.experts\.(\d+)\.", rn)
                    if m3 and int(m3.group(1)) >= N_EXP:
                        continue
                    tensors[nn] = torch.ones((), dtype=torch.float32)
                    continue
                m3 = re.match(r".*\.experts\.(\d+)\.", rn)
                if m3 and int(m3.group(1)) >= N_EXP:
                    continue
                t = f.get_tensor(rn)
                scale = None
                for cand in (rn + ".scale", rn.replace(".weight", ".scale"),
                             rn.replace(".weight", ".weight_scale")):
                    if cand in keys:
                        scale = f.get_tensor(cand)
                        break
                t = dequant_if_fp8(t, scale)
                s = apply_slice(role, role, t)
                del t
                if s is None:
                    pat = re.sub(r"\d+", "N", rn)
                    skipped[pat] = skipped.get(pat, 0) + 1
                    continue
                tensors[nn] = s
        del_keys = None  # keep memory flat; tensors dict holds slices only
    # nano scale placeholders for the names 01_synth emits as ones
    scale_names = [n for n in tensors if n.endswith(".scale")]
    report = {
        "step": "01b_synth_from_ablit",
        "real_tensors": len(wmap),
        "nano_tensors": len(tensors),
        "scale_names_found": len(scale_names),
        "skipped_patterns": dict(sorted(skipped.items())[:25]),
        "dry_run": not a.execute,
    }
    print(json.dumps(report, indent=1)[:4000])
    if not a.execute:
        return 0

    out = WD / "nano-dsv4-vision-ablit"
    out.mkdir(exist_ok=True)
    names = sorted(tensors)
    per = 800
    wmap_out = {}
    total = 0
    shards_n = 0
    for s0 in range(0, len(names), per):
        shards_n += 1
        fname = f"model-{shards_n:05d}-of-XXXXX.safetensors"
        chunk = {}
        for n in names[s0:s0 + per]:
            t = tensors[n]
            total += t.numel() * t.element_size()
            chunk[n] = t
        save_file(chunk, str(out / fname))
        for n in chunk:
            wmap_out[n] = fname
    for f in out.glob("model-*-of-XXXXX.safetensors"):
        f.rename(f.parent / f.name.replace("XXXXX", f"{shards_n:05d}"))
    wmap_out = {n: f.replace("XXXXX", f"{shards_n:05d}")
                for n, f in wmap_out.items()}
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": wmap_out}, indent=1))
    (out / "config.json").write_text(json.dumps(NANO_CFG, indent=1))
    print(json.dumps({"written": str(out), "shards": shards_n,
                      "tensors": len(names)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
