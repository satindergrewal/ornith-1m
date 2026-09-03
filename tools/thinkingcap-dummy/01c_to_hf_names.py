#!/usr/bin/env python3
"""ThinkingCap dummy - STEP 01c: convert nano to HF (transformers) names.

Scope: LANGUAGE layers only. transformers' deepseek_v4 class is text-only
(no MTP, no vision) - exactly the surface LoRA training (04) touches.
Vision/MTP/scales/hc/tid stay in the RAW checkpoint and re-merge later
(05) via tensor surgery; both A/B arms go through this same conversion
so they stay consistent.

Conventions learned from the 5.16.1 module tree:
  attn.wq_a/wq_b/wkv/wo_a/wo_b -> self_attn.q_a_proj/q_b_proj/kv_proj/
                                  o_a_proj/o_b_proj
  attn.q_norm/kv_norm          -> self_attn.q_a_norm/kv_a_norm
  attn.attn_sink               -> self_attn.sinks
  attn_norm/ffn_norm           -> input_layernorm/post_attention_layernorm
  ffn.gate.weight              -> mlp.gate.weight
  ffn.shared_experts w1/w2/w3  -> mlp.shared_experts gate/down/up_proj
  ffn.experts.E w1+w3 / w2     -> mlp.experts.gate_up_proj / down_proj
                                  (STACKED over experts)
Random-init whitelist (raw-side or derived params): hc_*, tid2eid,
scales, mtp.*, vision.*, aligner.*, image_*, hc_head_*.

Output: <wd>/nano-dsv4-vision-ablit-hf/ (single shard + tokenizer).
"""
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

WD = Path(os.environ.get("TC_WD", "/home/satinder/thinkingcap-dummy"))
SRC = WD / "nano-dsv4-vision-ablit"
OUT = WD / "nano-dsv4-vision-ablit-hf"

SKIP_PREFIXES = ("mtp.", "vision.", "aligner.", "image_", "hc_head_")


def lang_candidate(name):
    """HF name for one raw LANGUAGE-layer tensor, else None."""
    m = re.match(r"^layers\.(\d+)\.(.*)$", name)
    if not m:
        return None
    i, r = m.groups()
    P = f"model.layers.{i}."
    tab = {
        "attn.wq_a.weight": "self_attn.q_a_proj.weight",
        "attn.wq_b.weight": "self_attn.q_b_proj.weight",
        "attn.wkv.weight": "self_attn.kv_proj.weight",
        "attn.wo_a.weight": "self_attn.o_a_proj.weight",
        "attn.wo_b.weight": "self_attn.o_b_proj.weight",
        "attn.q_norm.weight": "self_attn.q_a_norm.weight",
        "attn.kv_norm.weight": "self_attn.kv_norm.weight",
        "attn.attn_sink": "self_attn.sinks",
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn.gate.weight": "mlp.gate.weight",
        "ffn.gate.bias": "mlp.gate.e_score_correction_bias",
        "ffn.gate.bias_vl": "mlp.gate.bias_vl",
        "ffn.shared_experts.w1.weight": "mlp.shared_experts.gate_proj.weight",
        "ffn.shared_experts.w2.weight": "mlp.shared_experts.down_proj.weight",
        "ffn.shared_experts.w3.weight": "mlp.shared_experts.up_proj.weight",
        "attn.compressor.wkv.weight": "self_attn.compressor.kv_proj.weight",
        "attn.compressor.wgate.weight": "self_attn.compressor.gate_proj.weight",
        "attn.compressor.norm.weight": "self_attn.compressor.kv_norm.weight",
        "attn.compressor.ape": "self_attn.compressor.position_bias",
        "attn.indexer.wq_b.weight":
            "self_attn.compressor.indexer.q_b_proj.weight",
        "attn.indexer.weights_proj.weight":
            "self_attn.compressor.indexer.scorer.weights_proj.weight",
        "attn.indexer.compressor.wkv.weight":
            "self_attn.compressor.indexer.kv_proj.weight",
        "attn.indexer.compressor.wgate.weight":
            "self_attn.compressor.indexer.gate_proj.weight",
        "attn.indexer.compressor.norm.weight":
            "self_attn.compressor.indexer.kv_norm.weight",
        "attn.indexer.compressor.ape":
            "self_attn.compressor.indexer.position_bias",
    }
    if r in tab:
        return P + tab[r]
    return None


def main():
    cfg = AutoConfig.from_pretrained(str(SRC))
    model = AutoModelForCausalLM.from_config(cfg)
    expected = set(model.state_dict().keys())
    del model

    idx = json.loads((SRC / "model.safetensors.index.json").read_text())
    names = sorted(idx["weight_map"])

    assign, used, skipped = {}, set(), []
    for n in names:
        if n.endswith(".scale") or n.startswith(SKIP_PREFIXES) \
                or ".hc_" in n or n.startswith("hc_") \
                or "ffn.gate.tid" in n or ".experts." in n:
            skipped.append(n)
            continue
        if n == "embed.weight":
            c = "model.embed_tokens.weight"
        elif n == "head.weight":
            c = "lm_head.weight"
        elif n == "norm.weight":
            c = "model.norm.weight"
        else:
            c = lang_candidate(n)
        if c is None or c in used or c not in expected:
            skipped.append(n)
            continue
        assign[n] = c
        used.add(c)

    # stacked experts per layer
    exp = {}
    for n in names:
        m = re.match(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w\d)\.weight$", n)
        if m:
            exp.setdefault((m.group(1), m.group(2)), {})[m.group(3)] = n
    stacked = {}
    for (li, ei), ws in exp.items():
        if set(ws) != {"w1", "w2", "w3"}:
            continue
        stacked.setdefault(li, []).append((int(ei), ws))
    for li, rows in stacked.items():
        rows.sort()
        gu = f"model.layers.{li}.mlp.experts.gate_up_proj"
        dn = f"model.layers.{li}.mlp.experts.down_proj"
        if gu in expected and dn in expected:
            assign[("STACK_GU", li)] = gu
            assign[("STACK_DN", li)] = dn
            used.add(gu); used.add(dn)

    missing = sorted(expected - used)
    wl = ("attn_hc.", "ffn_hc.", "hc_head.", "tid2eid")
    bad_missing = [k for k in missing if not any(w in k for w in wl)]
    print(json.dumps({"mapped": len(assign), "skipped_src": len(skipped),
                      "whitelisted_missing": len(missing) - len(bad_missing),
                      "bad_missing": bad_missing[:20],
                      "skipped_sample": skipped[:10]}, indent=1))
    if bad_missing:
        return 1

    tensors = {}
    with safe_open(str(SRC / "model-00001-of-00001.safetensors"),
                   framework="pt", device="cpu") as f:
        for key, hf in assign.items():
            if isinstance(key, tuple):
                continue
            tensors[hf] = f.get_tensor(key)
        for li, rows in stacked.items():
            w1 = torch.stack([f.get_tensor(ws["w1"]) for _, ws in rows])
            w3 = torch.stack([f.get_tensor(ws["w3"]) for _, ws in rows])
            w2 = torch.stack([f.get_tensor(ws["w2"]) for _, ws in rows])
            tensors[f"model.layers.{li}.mlp.experts.gate_up_proj"] = \
                torch.cat([w1, w3], dim=1)
            tensors[f"model.layers.{li}.mlp.experts.down_proj"] = w2

    OUT.mkdir(exist_ok=True)
    save_file(tensors, str(OUT / "model.safetensors"))
    (OUT / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(t.numel() * t.element_size()
                                         for t in tensors.values())},
         "weight_map": {n: "model.safetensors" for n in tensors}}, indent=1))
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (OUT / fname).write_bytes((SRC / fname).read_bytes())
    print("CONVERTED:", len(tensors), "tensors ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
