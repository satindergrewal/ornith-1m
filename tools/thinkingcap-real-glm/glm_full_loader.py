#!/usr/bin/env python3
"""glm_full_loader.py — index-driven streaming loader for the GLM-5.3-Flash
FP8 abliterated checkpoint (satgeze/GLM-5.3-Flash-ablit-work; 62 shards,
328GB, DeepSeek-style FP8-E4M3 with F32 weight_scale_inv on 128x128 blocks).

Mirrors full_loader.py (the DSV4 harness) so the engine, regen, trainer and
gate plug in with minimal deltas:
  * tensor_meta.json (built by glm_peek.py) -> np.memmap per shard, zero-copy
    slice, dequant ON DEVICE.
  * Codec: w[i,j] = LUT_e4m3(byte) * scale_inv[i//128, j//128]  (F32 linear
    scales — simpler than DSV4's E8M0 exponent scales: no exp2).
  * BF16 / F32 / I64 passthrough. KDA attention tensors are BF16 (the FP8
    checkpoint omits their scales: modules_to_not_convert attn_mha/attn_mqa);
    MoE experts + dense FFN + shared experts + DSA q/kv projections are FP8.
  * Streaming: load_layer(i) = every non-expert tensor of layer i on device;
    load_expert(i, e) = one routed expert (gate, up, down). visual.* and
    layers.45.* (MTP) are not in the meta at all (glm_peek skips them).
  * Selftest: two-way dequant (this LUT vs direct safetensors read +
    independent repeat_interleave block math) on a DSA projection, a dense
    FFN matrix and one routed expert."""

import json
import os
import sys

import numpy as np
import torch

MODEL_DIR = os.environ.get("GLM_MODEL", "/root/glm-fp8")
META_PATH = os.environ.get("GLM_META", "/root/tc-glm/tensor_meta.json")

P = "model.language_model."

_DT = {
    "BF16": (np.dtype("<u2"), torch.bfloat16),
    "F32": (np.dtype("<f4"), torch.float32),
    "I64": (np.dtype("<i8"), torch.int64),
    "F8_E4M3": (np.dtype("u1"), None),
}


def _e4m3_lut():
    """float8_e4m3fn decode table, 256 entries (same table as the DSV4 loader)."""
    b = np.arange(256, dtype=np.uint32)
    s = (b >> 7).astype(np.int32)
    e = ((b >> 3) & 15).astype(np.int32)
    m = (b & 7).astype(np.int32)
    val = np.where(e == 0, m.astype(np.float64) * 2.0 ** -9,
                   (1.0 + m / 8.0) * np.exp2((e - 7).astype(np.float64)))
    val = np.where((e == 15) & (m == 7), np.nan, val)
    val = np.where(s == 1, -val, val)
    return torch.from_numpy(val.astype(np.float32))


E4M3_LUT = _e4m3_lut()


class StreamingGLM:
    def __init__(self, model_dir=MODEL_DIR, meta_path=META_PATH, device="cuda"):
        self.dir = model_dir
        self.device = device
        d = json.load(open(meta_path))
        self.meta = d["meta"]
        self.offs = d["offsets"]
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.cfg = cfg["text_config"]
        self.n_layers = self.cfg["num_hidden_layers"]
        self.n_experts = self.cfg["n_routed_experts"]
        self._mm = {}
        self._lut8 = None

    # ------------------------------------------------------------------ raw io
    def _mmap(self, shard):
        mm = self._mm.get(shard)
        if mm is None:
            mm = np.memmap(os.path.join(self.dir, shard), dtype=np.uint8, mode="r")
            self._mm[shard] = mm
        return mm

    def _raw(self, name):
        dt, shape = self.meta[name][0], self.meta[name][1]
        shard, start, nbytes = self.offs[name]
        buf = np.frombuffer(self._mmap(shard)[start:start + nbytes],
                            dtype=_DT[dt][0])
        t = torch.from_numpy(buf.copy())
        if dt == "F8_E4M3":
            t = t.view(torch.uint8)
        return t.reshape(shape).to(self.device), dt, tuple(shape)

    def _l8(self):
        if self._lut8 is None:
            self._lut8 = E4M3_LUT.to(self.device)
        return self._lut8

    # ------------------------------------------------------------------ codecs
    def dequant_fp8(self, wname):
        """F8_E4M3 weight + F32 weight_scale_inv 128x128 block -> bf16 [n, k]."""
        sname = wname[: wname.rfind(".weight")] + ".weight_scale_inv"
        w, dt, (n, k) = self._raw(wname)
        assert dt == "F8_E4M3", (wname, dt)
        s, sdt, _ = self._raw(sname)
        assert sdt == "F32", (sname, sdt)
        wf = self._l8()[w.long()]                                # (n, k) f32
        sf = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
        return (wf * sf).to(torch.bfloat16)

    def passthrough(self, name, dtype=None):
        t, dt, shape = self._raw(name)
        tgt = dtype or _DT[dt][1]
        if dt == "BF16":
            t = t.view(torch.bfloat16)
        elif dt == "F32":
            t = t.view(torch.float32)
        elif dt == "I64":
            t = t.view(torch.int64)
        assert t.dtype == tgt, (name, t.dtype, tgt)
        return t.reshape(shape).contiguous()

    # ------------------------------------------------------------------ layer api
    def layer_tensor_names(self, i, experts=False):
        pre = f"{P}layers.{i}."
        out = []
        for k in self.meta:
            if not k.startswith(pre):
                continue
            if ".mlp.experts." in k and not experts:
                continue
            out.append(k)
        return sorted(out)

    def load_layer(self, i):
        """All non-expert tensors of layer i on device; short names (the
        'model.language_model.layers.{i}.' prefix stripped, checkpoint names
        otherwise kept verbatim so the engine cites them exactly)."""
        assert 0 <= i < self.n_layers
        out = {}
        for name in self.layer_tensor_names(i):
            if name.endswith(".weight_scale_inv"):
                continue                                       # consumed in dequant
            short = name[len(f"{P}layers.{i}."):]
            if self.meta[name][0] == "F8_E4M3":
                out[short] = self.dequant_fp8(name)
            else:
                out[short] = self.passthrough(name)
        return out

    def load_expert(self, i, e):
        """One routed expert -> (gate, up, down) bf16 on device."""
        p = f"{P}layers.{i}.mlp.experts.{e}."
        return (self.dequant_fp8(p + "gate_proj.weight"),
                self.dequant_fp8(p + "up_proj.weight"),
                self.dequant_fp8(p + "down_proj.weight"))

    def load_top(self):
        return {
            "embed": self.passthrough(P + "embed_tokens.weight"),
            "head": self.passthrough("lm_head.weight"),
            "norm": self.passthrough(P + "norm.weight"),
        }

    # ------------------------------------------------------------------ report
    def layer_bytes(self, i, experts=True):
        pre = f"{P}layers.{i}."
        tot = exp = 0
        for k, v in self.offs.items():
            if k.startswith(pre):
                tot += v[2]
                if ".mlp.experts." in k:
                    exp += v[2]
        return (tot, exp) if experts else tot


# ------------------------------------------------------------------ selftest

def _independent_fp8(loader, name):
    """Direct safetensors read + native float8 cast + block math (no LUT)."""
    from safetensors import safe_open
    sname = name[: name.rfind(".weight")] + ".weight_scale_inv"
    shard = os.path.join(loader.dir, loader.meta[name][2])
    with safe_open(shard, framework="pt") as f:
        w = f.get_tensor(name).float()
        s = f.get_tensor(sname).float()
    n, k = w.shape
    sb = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
    return (w * sb).to(torch.bfloat16).to(loader.device)


def selftest():
    L = StreamingGLM()
    print(f"[loader] layers={L.n_layers} experts={L.n_experts} device={L.device}")
    torch.cuda.init()

    # (a) FP8 two-way on a DSA q_a_proj (L3), a dense FFN matrix (L1) and an expert
    for name in (f"{P}layers.3.self_attn.q_a_proj.weight",
                 f"{P}layers.1.mlp.gate_proj.weight",
                 f"{P}layers.20.mlp.experts.7.up_proj.weight"):
        a = L.dequant_fp8(name)
        b = _independent_fp8(L, name)
        same = torch.equal(a, b)
        md = (a.float() - b.float()).abs().max().item()
        print(f"[test a] {name.split('language_model.')[-1]}: "
              f"{tuple(a.shape)} bit-equal={same} maxdiff={md:.3e} "
              f"absmean={a.float().abs().mean().item():.5f}")
        assert same or md == 0.0, f"FP8 dequant mismatch {name}"

    # (b) load_layer completeness for each layer class
    for i, tag in ((0, "dense"), (3, "DSA"), (7, "KDA"), (44, "last-KDA")):
        lay = L.load_layer(i)
        n_kda = sum(1 for k in lay if k.startswith("self_attn."))
        print(f"[test b] L{i:02d} {tag}: {len(lay)} tensors "
              f"({n_kda} self_attn), experts excluded="
              f"{all('.experts.' not in k for k in lay)}")
        del lay

    # (c) byte report
    tot_all = 0
    for i in range(L.n_layers):
        tot, exp = L.layer_bytes(i)
        tot_all += tot
        if i in (0, 3, 7, 20, 44):
            print(f"[bytes] layer {i:2d}: {tot/1e9:.3f} GB (experts {exp/1e9:.3f})")
    print(f"[bytes] all {L.n_layers} language layers: {tot_all/1e9:.2f} GB")

    top = L.load_top()
    print(f"[test c] top: embed {tuple(top['embed'].shape)}, "
          f"head {tuple(top['head'].shape)}, norm {tuple(top['norm'].shape)}")
    assert top["head"].shape == (154880, 4096)
    print("[loader] SELFTEST PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        raise SystemExit(selftest())
    print("usage: glm_full_loader.py test   (import StreamingGLM otherwise)")
    raise SystemExit(1)
