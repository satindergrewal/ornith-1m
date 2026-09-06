#!/usr/bin/env python3
"""dsv4_geometry.py - single source of DSV4 truth, read from the LIVE master.

Every constant is discovered from /model/config.json + tensor_meta.json +
the LoRA adapter at import time. Nothing is hardcoded except the expected
values, which assert fail-closed: if any drifts, the campaign cannot run
(wrong census would poison every seal downstream).

Expected geometry (verified 2026-09-07 against the packed master on
tc-cap-vol, config model_type deepseek_v4):
  43 main layers (0..42, ALL routed - no dense prefix; layers 0-2 hash-route
  via ffn.gate.tid2eid I64 [vocab,6], layers 3-42 learned router)
  256 routed experts/layer, top_k 6, routed_scaling_factor 1.5, norm_topk_prob
  moe_intermediate 2048, hidden 4096 -> logical expert shapes (2048,4096) x2
  + (4096,2048); every dim % 128 (hard codec requirement)
  3 DSpark MTP modules mtp.{0,1,2} (attn + own 256-expert ffn + markov_head
  on mtp.2) - NATIVE scope v1, never encoded
  LoRA cap adapter: 258 sites = 6/layer x 43 (attn.wq_b/wkv/wo_b +
  ffn.shared_experts.w1/w2/w3), all F8_E4M3, SCALE = 16/8
"""

import json
import os
import re

EXPECTED = {
    "model_type": "deepseek_v4",
    "architecture": "DeepseekV4ForCausalLM",
    "num_hidden_layers": 43,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 2048,
    "hidden_size": 4096,
    "num_nextn_predict_layers": 3,
    "routed_scaling_factor": 1.5,
    "norm_topk_prob": True,
}

# fold semantics (fold_cap_to_bf16.py v3.1 + tc_lora_train) - do not drift
LORA_SCALE = 16 / 8
SITE_PATTERN = re.compile(
    r"^layers\.(\d+)\.(attn\.(?:wq_b|wkv|wo_b)|ffn\.shared_experts\.w[123])\.weight$")

ROUTED = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")
MTP_ROUTED = re.compile(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")


class Geometry:
    """Discovered + validated geometry. Construct once, pass everywhere."""

    def __init__(self, model_dir, meta_path, lora_path=None):
        self.model_dir = model_dir
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.cfg = cfg
        for k, v in EXPECTED.items():
            if k == "architecture":
                assert v in cfg.get("architectures", []), \
                    f"architecture drift: {cfg.get('architectures')} want {v}"
            else:
                assert cfg.get(k) == v, f"config drift {k}: {cfg.get(k)} want {v}"
        self.n_layers = cfg["num_hidden_layers"]
        self.n_experts = cfg["n_routed_experts"]
        self.top_k = cfg["num_experts_per_tok"]
        self.hidden = cfg["hidden_size"]
        self.inter = cfg["moe_intermediate_size"]
        self.route_scale = cfg["routed_scaling_factor"]
        self.vocab = cfg["vocab_size"]

        meta = json.load(open(meta_path))["meta"]
        self.meta = meta

        # hash-routed layers are DISCOVERED from tid2eid presence, not assumed
        self.hash_layers = sorted(int(m.group(1)) for n in meta
                                  if "ffn.gate.tid2eid" in n
                                  for m in [re.match(r"^layers\.(\d+)\.", n)])
        assert self.hash_layers == [0, 1, 2], \
            f"hash layer drift: {self.hash_layers}"

        # routed census from the meta itself
        per_layer = {}
        for n in meta:
            m = ROUTED.match(n)
            if m:
                per_layer.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
        assert sorted(per_layer) == list(range(self.n_layers)), \
            "layer census drift"
        bad = {L: len(E) for L, E in per_layer.items()
               if len(E) != self.n_experts}
        assert not bad, f"expert census drift: {bad}"

        mtp_mods = sorted({int(m.group(1)) for n in meta
                          for m in [MTP_ROUTED.match(n)] if m})
        assert mtp_mods == [0, 1, 2], f"mtp census drift: {mtp_mods}"
        self.mtp_modules = mtp_mods

        # logical expert shapes: w1/w3 (inter, hidden) gate/up, w2 (hidden, inter) down
        n, k = self.inter, self.hidden
        self.logical_shapes = {"w1": (n, k), "w3": (n, k), "w2": (k, n)}
        for d in (n, k):
            assert d % 128 == 0, f"codec %128 violation: {d}"
        # engine roles (dsv4_full.swiglu): gate=x@w1.T silu'd, up=x@w3.T,
        # down=h@w2.T -> codec vocabulary gate_proj=w1, up_proj=w3, down_proj=w2
        self.role_map = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}

        # census constants the campaign arithmetic is derived from
        self.main_routed_tensors = self.n_layers * self.n_experts * 3
        self.mtp_routed_tensors = len(mtp_mods) * self.n_experts * 3

        self.lora_sites = None
        if lora_path:
            self.bind_lora(lora_path)

    def bind_lora(self, lora_path):
        from safetensors.torch import load_file
        raw = load_file(lora_path)
        sites = sorted({k[len("lora."):-2] for k in raw
                        if k.startswith("lora.") and k.endswith((".A", ".B"))})
        assert len(sites) == 6 * self.n_layers, \
            f"lora site census drift: {len(sites)} want {6 * self.n_layers}"
        for s in sites:
            assert SITE_PATTERN.match(s), f"unexpected lora site shape: {s}"
            assert self.meta.get(s, ["?"])[0] == "F8_E4M3", \
                f"lora site not F8_E4M3: {s}"
        self.lora_sites = set(sites)
        return sites


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("DSV4_MODEL", "/model"))
    ap.add_argument("--meta", default=os.environ.get("DSV4_META", "/wd/tensor_meta.json"))
    ap.add_argument("--lora", default="/wd/cap/lora.safetensors")
    a = ap.parse_args()
    g = Geometry(a.model_dir, a.meta, a.lora)
    print(f"[geom] OK: {g.n_layers} layers (hash {g.hash_layers}), "
          f"{g.n_experts} experts top{g.top_k} scale {g.route_scale}, "
          f"mtp {g.mtp_modules} native, "
          f"{g.main_routed_tensors} main routed tensors, "
          f"{len(g.lora_sites)} lora sites @ scale {LORA_SCALE}")


if __name__ == "__main__":
    main()
