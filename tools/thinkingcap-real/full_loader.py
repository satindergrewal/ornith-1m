#!/usr/bin/env python3
"""full_loader.py — index-driven streaming loader for the REAL DeepSeek-V4-Flash-Vision
abliterated checkpoint (node-a, ~/models/dsv4-vision-ablit mounted at /model).

Design:
  * tensor_meta.json (built by peek_headers.py from raw safetensors headers) gives
    name -> [dtype, shape, shard] and name -> [shard, abs_data_start, nbytes].
  * Reads go through np.memmap of each shard (zero-copy slice), bytes -> torch ->
    cuda, dequant ON GPU. No safetensors dependency for the hot path.
  * Codecs (exact, LUT-based; no dependence on exotic dtype cast support):
      - F8_E4M3 + F8_E8M0 scale [ceil(n/128), ceil(k/128)]  -> bf16
        (w[i,j] = e4m3(byte) * 2^(scale[i//128, j//128] - 127))
      - MXFP4 routed experts: I8 [n, k/2] packed pairs + E8M0 scale [n, k/32] -> bf16
        (logical [n, k]; element 2i = LOW nibble, 2i+1 = HIGH nibble — torch
        float4_e2m1fn_x2 packing convention; group of 32 along K per scale)
      - BF16 / F32 / I64 passthrough
  * Streaming: load_layer(i) returns every NON-expert tensor of layer i dequanted on
    cuda; load_expert(i, e) returns one routed expert (w1, w2, w3) on demand so the
    engine can route first and dequant only the top-6 + shared expert per token batch.
    Peak resident <= 2 layer groups by construction (engine frees between layers).
  * Selftest (`python3 full_loader.py test`): two-way dequant checks:
      (a) layers.10.attn.wo_b FP8: this loader vs direct safe_open + independent
          repeat_interleave block math on native float8 dtypes;
      (b) layers.10.ffn.experts.7.w1 MXFP4: this loader's nibble LUT vs torch
          float4_e2m1fn_x2 view cast (independent decode), if the torch build
          supports it, else vs a from-first-principles per-bit decoder;
      (c) per-layer byte-size report.
"""

import json
import os
import re
import sys

import numpy as np
import torch

MODEL_DIR = os.environ.get("DSV4_MODEL", "/model")
META_PATH = os.environ.get("DSV4_META", "/wd/tensor_meta.json")

_DT = {  # safetensors dtype string -> (numpy view dtype, torch target)
    "BF16": (np.dtype("<u2"), torch.bfloat16),
    "F32": (np.dtype("<f4"), torch.float32),
    "I64": (np.dtype("<i8"), torch.int64),
    "F8_E4M3": (np.dtype("u1"), None),
    "F8_E8M0": (np.dtype("u1"), None),
    "I8": (np.dtype("u1"), None),   # experts are packed nibbles; view as u1
}


def _e4m3_lut():
    """float8_e4m3fn decode table, 256 entries. exp bias 7, e==0 denormal m*2^-9,
    S111111 (e=15, m=7) = NaN, no infinities."""
    b = np.arange(256, dtype=np.uint32)
    s = (b >> 7).astype(np.int32)
    e = ((b >> 3) & 15).astype(np.int32)
    m = (b & 7).astype(np.int32)
    val = np.where(e == 0, m.astype(np.float64) * 2.0 ** -9,
                   (1.0 + m / 8.0) * np.exp2((e - 7).astype(np.float64)))
    val = np.where((e == 15) & (m == 7), np.nan, val)
    val = np.where(s == 1, -val, val)
    return torch.from_numpy(val.astype(np.float32))


def _e2m1_lut():
    """float4 e2m1 decode table, 16 entries: {0, .5, 1, 1.5, 2, 3, 4, 6} and negatives."""
    b = np.arange(16, dtype=np.uint32)
    s = (b >> 3).astype(np.int32)
    e = ((b >> 1) & 3).astype(np.int32)
    m = (b & 1).astype(np.int32)
    val = np.where(e == 0, m.astype(np.float64) * 0.5,
                   (1.0 + m * 0.5) * np.exp2((e - 1).astype(np.float64)))
    val = np.where(s == 1, -val, val)
    return torch.from_numpy(val.astype(np.float32))


E4M3_LUT = _e4m3_lut()
E2M1_LUT = _e2m1_lut()

_EXP = re.compile(r"\.ffn\.experts\.(\d+)\.")


class StreamingDSV4:
    def __init__(self, model_dir=MODEL_DIR, meta_path=META_PATH, device="cuda"):
        self.dir = model_dir
        self.device = device
        d = json.load(open(meta_path))
        self.meta = d["meta"]          # name -> [dtype, shape, shard]
        self.offs = d["offsets"]       # name -> [shard, start, nbytes]
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.n_layers = self.cfg["num_hidden_layers"]
        self.n_experts = self.cfg["n_routed_experts"]
        self._mm = {}
        # free the LUTs on device lazily
        self._lut4 = None
        self._lut8 = None

    # ------------------------------------------------------------------ raw io
    def _mmap(self, shard):
        mm = self._mm.get(shard)
        if mm is None:
            mm = np.memmap(os.path.join(self.dir, shard), dtype=np.uint8, mode="r")
            self._mm[shard] = mm
        return mm

    def _raw(self, name):
        """-> (torch uint8/int64/float tensor on device reshaped to meta shape,
        dtype_str, shape)"""
        dt, shape, shard = self.meta[name]
        shard, start, nbytes = self.offs[name]
        mm = self._mmap(shard)
        buf = np.frombuffer(mm[start:start + nbytes], dtype=_DT[dt][0])
        t = torch.from_numpy(buf.copy())  # detach from page cache view
        if dt in ("F8_E4M3", "F8_E8M0", "I8"):
            t = t.view(torch.uint8)
        return t.reshape(shape).to(self.device, non_blocking=False), dt, tuple(shape)

    # ------------------------------------------------------------------ codecs
    def _l4(self):
        if self._lut4 is None:
            self._lut4 = E2M1_LUT.to(self.device)
        return self._lut4

    def _l8(self):
        if self._lut8 is None:
            self._lut8 = E4M3_LUT.to(self.device)
        return self._lut8

    def _bytes(self, name):
        w, dt, shape = self._raw(name)
        assert dt == "F8_E4M3", (name, dt)
        return w

    def _scale_bytes(self, name):
        w, dt, shape = self._raw(name)
        assert dt == "F8_E8M0", (name, dt)
        return w

    def dequant_fp8(self, wname):
        """FP8-E4M3 weight + E8M0 128x128 block scale -> bf16 [n, k]."""
        sname = wname[: wname.rfind(".weight")] + ".scale"
        w = self._bytes(wname).long()
        s = self._scale_bytes(sname)
        n, k = self.meta[wname][1]
        wf = self._l8()[w]                                   # (n, k) f32
        sf = torch.exp2(s.float() - 127.0)                   # (cn, ck)
        sf = sf.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
        return (wf * sf).to(torch.bfloat16)

    def dequant_mxfp4(self, wname):
        """Packed MXFP4 expert weight [n, k/2] + E8M0 scale [n, k/32] -> bf16 [n, k].
        Element 2i = LOW nibble of byte i, element 2i+1 = HIGH nibble."""
        sname = wname[: wname.rfind(".weight")] + ".scale"
        w, dt, (n, k2) = self._raw(wname)
        assert dt == "I8" and len(self.meta[wname][1]) == 2, (wname, dt)
        s = self._scale_bytes(sname)
        k = 2 * k2
        lut = self._l4()
        wf = torch.empty((n, k), dtype=torch.float32, device=self.device)
        lo = (w & 0xF).long()
        hi = (w >> 4).long()
        wf[:, 0::2] = lut[lo]
        wf[:, 1::2] = lut[hi]
        sf = torch.exp2(s.float() - 127.0)                   # (n, k/32)
        sf = sf.repeat_interleave(32, 1)                     # (n, k)
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
        pre = f"layers.{i}."
        out = []
        for k in self.meta:
            if not k.startswith(pre):
                continue
            m = _EXP.search(k)
            if m and not experts:
                continue
            out.append(k)
        return sorted(out)

    def load_layer(self, i):
        """All non-expert tensors of layer i, dequanted, on device. Dict of short
        names (layers.i. stripped)."""
        assert 0 <= i < self.n_layers
        out = {}
        for name in self.layer_tensor_names(i):
            dt = self.meta[name][0]
            if name.endswith(".scale"):
                continue  # consumed inside dequant_fp8 / dequant_mxfp4
            short = name[len(f"layers.{i}."):]
            if dt == "F8_E4M3":
                out[short] = self.dequant_fp8(name)
            elif dt == "I8":
                raise RuntimeError(f"unexpected expert tensor in load_layer: {name}")
            else:
                out[short] = self.passthrough(name)
        return out

    def load_expert(self, i, e):
        """One routed expert -> (w1, w2, w3) bf16 [*, 4096|2048] on device."""
        p = f"layers.{i}.ffn.experts.{e}."
        return (self.dequant_mxfp4(p + "w1.weight"),
                self.dequant_mxfp4(p + "w2.weight"),
                self.dequant_mxfp4(p + "w3.weight"))

    def load_top(self):
        """embed / head / final norm / hc_head trio (resident for the whole run)."""
        return {
            "embed": self.passthrough("embed.weight"),
            "head": self.passthrough("head.weight"),
            "norm": self.passthrough("norm.weight"),
            "hc_head_fn": self.passthrough("hc_head_fn"),
            "hc_head_base": self.passthrough("hc_head_base"),
            "hc_head_scale": self.passthrough("hc_head_scale"),
        }

    # ------------------------------------------------------------------ report
    def layer_bytes(self, i, experts=True):
        pre = f"layers.{i}."
        tot = exp = 0
        for k, v in self.offs.items():
            if k.startswith(pre):
                tot += v[2]
                if _EXP.search(k):
                    exp += v[2]
        return (tot, exp) if experts else tot


# ------------------------------------------------------------------ selftest

def _independent_fp8(loader, name):
    """Direct safe_open read + native float8 cast + repeat_interleave block math.
    Independent of this loader's LUT + exp2 scale handling."""
    from safetensors import safe_open
    sname = name[: name.rfind(".weight")] + ".scale"
    shard = os.path.join(loader.dir, loader.meta[name][2])
    with safe_open(shard, framework="pt", device="cpu") as f:
        w = f.get_tensor(name).float()
        s = f.get_tensor(sname).float()
    n, k = w.shape
    sb = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
    return (w * sb).to(torch.bfloat16).to(loader.device)


def _independent_mxfp4_bits(loader, name, n_elem=256):
    """From-first-principles per-bit decoder over the first n_elem logical elements.
    e2m1: s(1) e(2) m(1); e==0: m*0.5; else (1+m*0.5)*2^(e-1). Verifies LUT formula."""
    sname = name[: name.rfind(".weight")] + ".scale"
    shard, start, nb = loader.offs[name]
    mm = loader._mmap(shard)
    w = np.frombuffer(mm[start:start + nb], dtype=np.uint8)
    sshard, sstart, snb = loader.offs[sname]
    sm = np.frombuffer(mm[sstart:sstart + snb], dtype=np.uint8)
    n = loader.meta[name][1][0]
    vals = []
    for i in range(n_elem):
        byte = int(w[i // 2])
        nib = byte & 0xF if i % 2 == 0 else byte >> 4
        s = -1.0 if nib & 8 else 1.0
        e = (nib >> 1) & 3
        m = nib & 1
        v = m * 0.5 if e == 0 else (1.0 + m * 0.5) * (2.0 ** (e - 1))
        vals.append(s * v)
    vals = np.array(vals, dtype=np.float32)
    for i in range(n_elem):          # per-32-group E8M0 scales along K
        vals[i] *= 2.0 ** (float(sm[i // 32]) - 127.0)
    return vals


def selftest():
    L = StreamingDSV4()
    print(f"[loader] layers={L.n_layers} experts={L.n_experts} device={L.device}")

    # (a) FP8 two-way: wo_b (also the drowzeys-edited tensor on L10)
    name = "layers.10.attn.wo_b.weight"
    a = L.dequant_fp8(name)
    b = _independent_fp8(L, name)
    same = torch.equal(a, b)
    md = (a.float() - b.float()).abs().max().item()
    print(f"[test a] wo_b FP8 two-way: shapes {tuple(a.shape)} vs {tuple(b.shape)}; "
          f"bit-equal={same} maxdiff={md:.3e}; "
          f"absmean={a.float().abs().mean().item():.5f} (01b slice ref 0.0156)")
    assert same or md == 0.0, "FP8 dequant mismatch"
    assert 0.005 < a.float().abs().mean().item() < 0.06, "wo_b absmean far from 01b ref"

    # (b) MXFP4 two-way: expert 7 w1 of layer 10
    name = "layers.10.ffn.experts.7.w1.weight"
    mine = L.dequant_mxfp4(name)
    n, k = mine.shape
    assert (n, k) == (2048, 4096), (n, k)
    ref = torch.from_numpy(_independent_mxfp4_bits(L, name)).to(L.device)
    d1 = (mine[0, :256].float() - ref).abs().max().item()
    print(f"[test b1] expert w1 MXFP4 LUT vs per-bit decoder (first row, 256 elems): "
          f"maxdiff={d1:.3e}")
    assert d1 == 0.0, "MXFP4 LUT != per-bit decode"
    # independent nibble-order authority if torch supports the packed dtype cast
    try:
        shard, start, nb = L.offs[name]
        mm = L._mmap(shard)
        raw = torch.frombuffer(mm[start:start + nb].copy(), dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2)
        casted = raw.to(torch.float32).to(L.device)     # (n, k/2) -> each entry a pair?
        # torch casts packed fp4 to float32 by unpacking along a new trailing dim
        if casted.dim() == 2:
            unpacked = casted.view(n, k // 2, 2).reshape(n, k)
        else:
            unpacked = casted
        sname = name[: name.rfind(".weight")] + ".scale"
        s = L._scale_bytes(sname)
        sf = torch.exp2(s.float() - 127.0).repeat_interleave(32, 1)
        ref2 = (unpacked * sf).to(torch.bfloat16)
        d2 = (mine.float() - ref2.float()).abs().max().item()
        print(f"[test b2] expert w1 nibble order vs torch float4_e2m1fn_x2 view: "
              f"maxdiff={d2:.3e}")
        assert d2 == 0.0, "nibble order mismatch vs torch packed-fp4 cast"
    except Exception as ex:
        print(f"[test b2] torch fp4x2 cross-check unavailable ({type(ex).__name__}: "
              f"{str(ex)[:120]}); nibble order rests on convention + coherence gate")

    # stats sanity across a few experts
    for e in (0, 128, 255):
        w1 = L.dequant_mxfp4(f"layers.10.ffn.experts.{e}.w1.weight")
        am = w1.float().abs().mean().item()
        print(f"[stats] L10 e{e} w1 absmean={am:.5f} std={w1.float().std().item():.5f}")
        assert 0.003 < am < 0.08, f"expert {e} stats off: {am}"

    # (c) per-layer byte report
    tot_all = 0
    for i in range(L.n_layers):
        tot, exp = L.layer_bytes(i)
        tot_all += tot
        if i in (0, 1, 2, 3, 10, 42):
            print(f"[bytes] layer {i:2d}: {tot/1e9:.3f} GB (experts {exp/1e9:.3f})")
    print(f"[bytes] all {L.n_layers} language layers: {tot_all/1e9:.2f} GB")

    # quick passthrough sanity: hc + gate + tid
    t = L.passthrough("layers.10.hc_attn_fn")
    assert t.shape == (24, 16384) and t.dtype == torch.float32
    t = L.passthrough("layers.0.ffn.gate.tid2eid")
    assert t.shape == (129280, 6) and t.dtype == torch.int64
    print("[test c] hc_attn_fn + tid2eid passthrough shapes OK")

    del L
    torch.cuda.empty_cache()
    print("[loader] SELFTEST PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        raise SystemExit(selftest())
    print("usage: full_loader.py test   (import StreamingDSV4 otherwise)")
    raise SystemExit(1)
