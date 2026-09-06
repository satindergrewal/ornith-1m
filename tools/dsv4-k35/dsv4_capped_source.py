#!/usr/bin/env python3
"""dsv4_capped_source.py - sealed dequant-on-demand source for the encode.

Replaces the 310GB BF16 fold: the encode reads the packed FP8/MXFP4 master
directly, dequants + folds the cap LoRA at load time. Numerically identical
to fold_cap_to_bf16.py v3.1 by construction - same dispatch, same fp32
accumulate, same single bf16 downcast - and the parity gate
(dsv4_source_parity.py) proves it bitwise against the 6 shards the fold
wrote before it died.

Artifact-native policy (v1, one rule): every non-expert tensor materializes
as its dequanted/folded value - F8_E4M3 -> bf16 (folded at the 258 sites),
BF16 -> bf16, F32 -> f32 kept, I64 -> raw. Expert tensors (load_expert)
have no LoRA sites. The mtp.* namespace is native scope, byte-derived the
same way, never encoded v1.
"""

import hashlib
import os

import torch

from full_loader import StreamingDSV4

SCALE = 16 / 8  # ALPHA / R from tc_lora_train (do not drift)


class CappedSource:
    def __init__(self, model_dir=None, meta_path=None, lora_path=None,
                 device="cuda:0"):
        self.L = StreamingDSV4(
            model_dir=model_dir or os.environ.get("DSV4_MODEL", "/model"),
            meta_path=meta_path or os.environ.get("DSV4_META", "/wd/tensor_meta.json"),
            device=device)
        from safetensors.torch import load_file
        lora_path = lora_path or "/wd/cap/lora.safetensors"
        raw = load_file(lora_path)
        lora = {}
        for k, v in raw.items():
            if not (k.startswith("lora.") and k.endswith((".A", ".B"))):
                continue
            site = k[len("lora."):-2]
            d = lora.setdefault(site, {})
            d[k[-1]] = v.float()
        unpaired = sorted(s for s, v in lora.items() if not ("A" in v and "B" in v))
        if unpaired:
            raise SystemExit(f"[src] ABORT unpaired A/B: {unpaired[:5]}")
        if not lora:
            raise SystemExit("[src] ABORT: no LoRA sites parsed")
        self.lora = {k: (v["A"], v["B"]) for k, v in lora.items()}

        non_scale = {n for n, (dt, _, _) in self.L.meta.items() if dt != "F8_E8M0"}
        ghost = sorted(set(self.lora) - non_scale)
        if ghost:
            raise SystemExit(f"[src] ABORT ghost sites: {ghost[:5]}")
        self.identity = {
            "lora_sha256": hashlib.sha256(open(lora_path, "rb").read()).hexdigest(),
            "lora_scale": SCALE,
            "lora_sites": len(self.lora),
        }
        print(f"[src] {len(self.lora)} sites, scale {SCALE}, "
              f"lora sha {self.identity['lora_sha256'][:12]}", flush=True)

    # ------------------------------------------------------------- experts
    def load_expert(self, layer, expert, mtp=None):
        """-> (gate, down, up) bf16, logical shapes, from the packed master.
        Expert tensors carry no LoRA sites, so this is pure dequant."""
        if mtp is not None:
            name = f"mtp.{mtp}.ffn.experts.{expert}"
            w1 = self.L.dequant_mxfp4(name + ".w1.weight")
            w2 = self.L.dequant_mxfp4(name + ".w2.weight")
            w3 = self.L.dequant_mxfp4(name + ".w3.weight")
        else:
            w1, w2, w3 = self.L.load_expert(layer, expert)
        return w1, w2, w3  # engine order: gate, down, up

    def expert_tensor(self, name):
        """-> single dequanted bf16 expert tensor by full meta name."""
        return self.L.dequant_mxfp4(name)

    # ------------------------------------------------------------- natives
    def load_native(self, name):
        """-> tensor exactly as fold_cap_to_bf16 wrote it (dtype included).
        Dispatch mirrors the fold verbatim; any unhandled dtype aborts."""
        dt = self.L.meta[name][0]
        if dt == "F8_E4M3":
            w = self.L.dequant_fp8(name).float()
            if name in self.lora:
                w = self._fold(name, w)
            return w.to(torch.bfloat16)
        if dt == "I8":
            return self.L.dequant_mxfp4(name).float().to(torch.bfloat16)
        if dt == "BF16":
            w = self.L.passthrough(name).float()
            if name in self.lora:  # defensive: sites are F8_E4M3 today
                w = self._fold(name, w)
            return w.to(torch.bfloat16)
        if dt == "F32":
            return self.L.passthrough(name).float()
        if dt == "I64":
            raw, _, _ = self.L._raw(name)
            return raw
        raise SystemExit(f"[src] unhandled dtype {dt} for {name}")

    def _fold(self, name, w):
        A, B = self.lora[name]
        delta = (B.to(w.device) @ A.to(w.device)) * SCALE
        if delta.shape != w.shape:
            raise SystemExit(f"[src] SHAPE MISMATCH {name}: "
                             f"W {tuple(w.shape)} vs delta {tuple(delta.shape)}")
        return w + delta

    # ------------------------------------------------------------- identity
    def native_names(self):
        """All non-expert, non-scale tensor names (fold output domain)."""
        out = []
        for n, (dt, _, _) in self.L.meta.items():
            if dt == "F8_E8M0":
                continue
            from dsv4_geometry import ROUTED, MTP_ROUTED
            if ROUTED.match(n) or MTP_ROUTED.match(n):
                continue
            out.append(n)
        return out
