#!/usr/bin/env python3
"""dsv4_full.py — full-model DeepSeek-V4-Flash engine on the real abliterated
checkpoint, layer-streamed for 128GB-UMA DGX Spark.

Generalizes tools/thinkingcap-dummy/nano_torch.py to the real config, with every
dummy-degenerate site replaced by the official semantics. Math authority: the
checkpoint's own reference implementation (inference/model.py + kernel.py, pure
torch), cross-checked against exllamav3 modules/dsv4.py + hyperconnections.py +
util/rope.py fetched from node-b. Line-cited notes below.

Config-driven: layer count, dims, experts, ratios all parsed from config.json.
No hardcoded shapes beyond config parsing.

Step-1 simplifications (documented PROGRESS.md ASSUMPTIONS #3):
  - QAT activation sims skipped (fp8-per-128 act sim on FP8-weight linear inputs,
    fp8-per-64 sim on kv nope dims, Hadamard+fp4 sim in the indexer). Weights are
    dequantized EXACTLY; matmuls run bf16/fp32. At smoke lengths (T = S//4 << 512)
    the indexer is dense regardless of scores.
  - MTP / vision / DSpark not loaded (step-2 surface).

Env knobs:
  HC_COMB_TRANSPOSED=1   use exl3's comb^T in hc_post (fallback A/B; default = official)
"""

import math
import os
import time

import torch
import torch.nn.functional as F

from full_loader import StreamingDSV4

HC_COMB_TRANSPOSED = os.environ.get("HC_COMB_TRANSPOSED", "0") == "1"


# ----------------------------------------------------------------- primitives

def rms_w(x, w, eps):
    """Weighted RMSNorm, fp32 compute, returns x.dtype (official RMSNorm.forward)."""
    d = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (w.float() * xf).to(d)


def rms_u(x, eps):
    """Unweighted per-head norm on the full trailing dim (official Attention q norm)."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def rot_pairs(x, cos, sin):
    """GPT-J adjacent-pair rotation on a (..., rd) slice. cos/sin broadcastable
    (..., rd/2). Mirrors apply_rotary_emb (complex-mult on (even, odd) pairs):
    fp32 math, returns x.dtype (reference stores back in place)."""
    d = x.dtype
    e, o = x[..., 0::2].float(), x[..., 1::2].float()
    r = torch.stack((e * cos - o * sin, o * cos + e * sin), -1).flatten(-2)
    return r.to(d)


def rope_cs(invf, pos):
    th = pos.float()[:, None] * invf[None, :]
    return th.cos(), th.sin()


def yarn_inv_freq(dim, base, scaling):
    """Official precompute_freqs_cis ramp (floor/ceil clamp, interpolate low freqs)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    factor = float(scaling["factor"])
    orig = int(scaling["original_max_position_embeddings"])
    bf = float(scaling.get("beta_fast", 32))
    bs = float(scaling.get("beta_slow", 1))

    def cdim(rot):
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))

    low, high = math.floor(cdim(bf)), math.ceil(cdim(bs))
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    smooth = 1 - ramp
    return freqs / factor * (1 - smooth) + freqs * smooth


# ----------------------------------------------------------------- engine

class DSV4Full:
    def __init__(self, loader: StreamingDSV4, load_top=True):
        self.L = loader
        c = loader.cfg
        self.cfg = c
        self.dev = loader.device
        self.n_layers = c["num_hidden_layers"]
        self.H = c["hidden_size"]
        self.n_heads = c["num_attention_heads"]
        self.hd = c["head_dim"]                       # 512, rope INCLUDED (trailing 64)
        self.rd = c["qk_rope_head_dim"]               # 64
        self.nope = self.hd - self.rd                 # 448
        self.qlr = c["q_lora_rank"]
        self.olr = c["o_lora_rank"]
        self.og = c["o_groups"]
        self.hpg = self.n_heads // self.og
        self.eps = c["rms_norm_eps"]
        self.window = c["sliding_window"]
        self.topk_idx = c["index_topk"]
        self.idx_h, self.idx_d = c["index_n_heads"], c["index_head_dim"]
        self.hc = c["hc_mult"]
        self.hc_iters, self.hc_eps = c["hc_sinkhorn_iters"], c["hc_eps"]
        self.n_exp, self.exp_k = c["n_routed_experts"], c["num_experts_per_tok"]
        self.route_scale = c["routed_scaling_factor"]
        self.swiglu_limit = c["swiglu_limit"]
        self.hash_layers = c.get("num_hash_layers", 0)
        self.eos = c.get("eos_token_id", 1)
        self.ratios = c["compress_ratios"][: self.n_layers]
        self.scale = self.hd ** -0.5                  # official softmax_scale

        # rope tables: sliding layers -> plain theta; csa/hca -> yarn on compress theta
        self.invf_main = 1.0 / c["rope_theta"] ** (
            torch.arange(0, self.rd, 2).float() / self.rd)
        self.invf_comp = yarn_inv_freq(self.rd, c["compress_rope_theta"], c["rope_scaling"])
        self.invf_main = self.invf_main.to(self.dev)
        self.invf_comp = self.invf_comp.to(self.dev)

        self.top = loader.load_top() if load_top else None
        if self.top is not None:
            self.top["head_f32"] = self.top["head"].float()   # logits in fp32

        # activation dtype: bf16 for inference (validated smoke path); the LoRA
        # trainer sets fp32 (exact adapter-delta gradients; tf32 matmuls allowed).
        self.act_dtype = torch.bfloat16
        # LoRA state: None (base), or {(layer, site): (A, B)} + scale. Delta path
        # (y = x@W.T + scale*((x@A.T)@B.T)) for training; folded mode applies
        # W_eff = W + scale*B@A at layer-load time (05-merge semantics, for the
        # capped gate arm).
        self.lora = None
        self.lora_scale = 1.0
        self.lora_fold = False

    # ---------------------------------------------------------------- helpers
    def attach_lora(self, path, scale, fold=False):
        """Load lora.safetensors (raw checkpoint names, dummy 04 format:
        lora.{layers.i.site.weight}.A / .B). fold=True applies the delta INTO the
        loaded bf16 weights (gate arm); fold=False keeps the separate delta path
        (training; bf16 rounding would otherwise swallow the small deltas)."""
        from safetensors.torch import load_file
        t = load_file(path)
        self.lora = {}
        self.lora_scale = scale
        self.lora_fold = fold
        for k in t:
            if not k.startswith("lora.") or not (k.endswith(".A") or k.endswith(".B")):
                continue
            site = k[len("lora."):-2]          # layers.i.site.weight
            li = int(site.split(".")[1])
            site = site.split(".", 2)[2]       # strip "layers.{i}." -> _lmm key format
            key = (li, site)
            d = self.lora.setdefault(key, {})
            d[k[-1]] = t[k].to(self.dev).float()
        self.lora = {k: (v["A"], v["B"]) for k, v in self.lora.items() if "A" in v and "B" in v}
        # pre-cast adapters to compute dtype: avoids per-step x.float() copies
        # (bf16 matmul accumulates in fp32 internally; rank-8 inner dim keeps
        # delta accuracy ~fp32 at zero cast cost)
        if not fold:
            self.lora = {k: (A.to(torch.bfloat16), B.to(torch.bfloat16))
                         for k, (A, B) in self.lora.items()}
        return len(self.lora)

    def _lmm(self, x, W, i, site):
        """Linear with optional LoRA delta path. Base path is EXACTLY x @ W.T
        (lora None / site absent) — the validated smoke math."""
        y = x @ W.to(x.dtype).T
        if self.lora is not None and not self.lora_fold:
            ab = self.lora.get((i, site))
            if ab is not None:
                A, B = ab
                y = y + ((x @ A.T) @ B.T) * self.lora_scale
        return y

    def _load_layer(self, i):
        L = self.L.load_layer(i)
        if self.lora is not None and self.lora_fold:
            for (li, site), (A, B) in self.lora.items():
                if li == i and site in L:
                    L[site] = (L[site].float()
                               + self.lora_scale * (B @ A)).to(torch.bfloat16)
        return L

    def _invf(self, ratio):
        return self.invf_main if ratio == 0 else self.invf_comp

    def _cos_sin(self, invf, pos, extra=None):
        cos, sin = rope_cs(invf, pos)
        if extra is None:
            return cos, sin
        return cos.view(*extra, -1), sin.view(*extra, -1)

    # ---------------------------------------------------------------- attention
    def attention(self, i, L, x):
        """x (S, H) bf16 normed -> (S, H) bf16. Official Attention.forward, start_pos=0."""
        S = x.shape[0]
        ratio = self.ratios[i]
        invf = self._invf(ratio)
        pos = torch.arange(S, device=self.dev)
        coss, sins = self._cos_sin(invf, pos)                       # (S, 32)

        # q path
        qr = rms_w(x @ L["attn.wq_a.weight"].to(x.dtype).T, L["attn.q_norm.weight"], self.eps)
        q = self._lmm(qr, L["attn.wq_b.weight"], i,
                       "attn.wq_b.weight").view(S, self.n_heads, self.hd)
        q = rms_u(q, self.eps)                                      # full-head unweighted
        cq, sq = coss.view(S, 1, self.rd // 2), sins.view(S, 1, self.rd // 2)
        q = torch.cat([q[..., : self.nope], rot_pairs(q[..., self.nope:], cq, sq)], -1)

        # kv path (kv_norm weighted over the FULL 512, then rope last 64)
        kv = rms_w(self._lmm(x, L["attn.wkv.weight"], i, "attn.wkv.weight"),
                   L["attn.kv_norm.weight"], self.eps)
        kv = torch.cat([kv[:, : self.nope], rot_pairs(kv[:, self.nope:], coss, sins)], -1)

        # window indices (causal sliding 128 within the chunk)
        base = pos[:, None]
        w = min(S, self.window)
        win = (base - self.window + 1).clamp(0) + torch.arange(w, device=self.dev)[None, :]
        win = torch.where(win > base, torch.full_like(win, -1), win)   # (S, w)

        parts = [win]
        if ratio:
            ent = self.compress(L, x, ratio, overlapping=(ratio == 4), indexer=False)
            T = ent.shape[0]
            if ratio == 4:  # CSA: indexer selection (topk over pool)
                sel = self.indexer_topk(i, L, x, qr, ent.shape[0])     # (S, k) with -1s
            else:           # HCA: dense causal bound
                mat = torch.arange(T, device=self.dev)[None, :].repeat(S, 1)
                bound = ((pos + 1) // ratio)[:, None]
                sel = torch.where(mat >= bound, torch.full_like(mat, -1), mat + S)
            kv_all = torch.cat([kv, ent], 0)
            parts.append(sel)
        else:
            kv_all = kv
        idx = torch.cat(parts, -1)                                   # (S, K)

        o = self.sparse_attn(q, kv_all, L["attn.attn_sink"], idx)     # (S, h, hd) rotated

        # eq.26 conjugate de-rotation at the query positions
        o = torch.cat([o[..., : self.nope],
                       rot_pairs(o[..., self.nope:], cq, -sq)], -1)

        # grouped o_proj: heads grouped 8x8; wo_a viewed (G, R, group_width)
        og = o.reshape(S, self.og, self.hpg * self.hd)
        woa = L["attn.wo_a.weight"].to(og.dtype).view(self.og, self.olr, self.hpg * self.hd)
        y = torch.einsum("sgd,grd->sgr", og, woa).reshape(S, self.og * self.olr)
        return self._lmm(y, L["attn.wo_b.weight"], i, "attn.wo_b.weight")

    def sparse_attn(self, q, kv, sinks, idx):
        """One softmax over [selected window+pool logits ++ per-head sink]; sink mass
        dropped. q (S,h,hd) bf16 (roped); kv (N,hd) bf16; idx (S,K) long, -1 skip."""
        S, h, d = q.shape
        valid = idx >= 0
        kvg = kv[idx.clamp(min=0)]                                   # (S,K,hd)
        logits = torch.einsum("shd,skd->shk", q.float(), kvg.float()) * self.scale
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        sink = sinks.float().view(1, h, 1)
        allg = torch.cat([logits, sink.expand(S, h, 1)], -1)
        p = torch.softmax(allg, -1)[..., :-1]                        # drop sink mass
        o = torch.einsum("shk,skd->shd", p, kvg.float())
        return o.to(self.act_dtype)

    # ---------------------------------------------------------------- compressor
    def compress(self, L, x, ratio, overlapping, indexer=False):
        """Official Compressor.forward, start_pos=0, stateless. Returns (T, hd or idx_d)
        bf16 entries roped at positions w*ratio. Remainder discarded."""
        p = "attn.indexer.compressor." if indexer else "attn.compressor."
        S = x.shape[0]
        T = S // ratio
        if T == 0:
            return torch.zeros((0, self.hd if not indexer else self.idx_d),
                               dtype=torch.bfloat16, device=self.dev)
        xf = x.float()
        wkv = L[p + "wkv.weight"].float()
        wg = L[p + "wgate.weight"].float()
        kv = xf @ wkv.T                                              # (S, W)
        g = xf @ wg.T
        W = kv.shape[-1]
        kv = kv[: T * ratio].view(T, ratio, W)
        g = g[: T * ratio].view(T, ratio, W) + L[p + "ape"].float()  # ape (ratio, W)
        if overlapping:
            d = W // 2
            nkv = kv.new_zeros((T, 2 * ratio, d))
            nkv[:, ratio:] = kv[..., d:]
            nkv[1:, :ratio] = kv[:-1, :, :d]
            ng = g.new_full((T, 2 * ratio, d), float("-inf"))
            ng[:, ratio:] = g[..., d:]
            ng[1:, :ratio] = g[:-1, :, :d]
            kv, g = nkv, ng
        ent = (kv * torch.softmax(g, dim=1)).sum(1)                  # (T, d)
        ent = rms_w(ent.to(self.act_dtype), L[p + "norm.weight"], self.eps)
        cosp, sinp = self._cos_sin(self.invf_comp,
                                   torch.arange(T, device=self.dev) * ratio)
        d_tot = ent.shape[-1]
        out = torch.cat([ent[:, : d_tot - self.rd],
                         rot_pairs(ent[:, d_tot - self.rd:], cosp, sinp)], -1)
        return out

    # ---------------------------------------------------------------- indexer
    def indexer_topk(self, i, L, x, qr, T, offset=None):
        """Official Indexer.forward (start_pos=0), QAT sims skipped. Returns (S, k)
        indices into the concat [window kv ++ pool] space (-1 sentinel).
        offset: base index of this sequence's pool inside kv_all (default S =
        single-sequence pack)."""
        S = x.shape[0]
        ratio = self.ratios[i]
        if offset is None:
            offset = S
        pos = torch.arange(S, device=self.dev)
        cq, sq = self._cos_sin(self.invf_comp, pos, (S, 1))
        q = (qr @ L["attn.indexer.wq_b.weight"].to(qr.dtype).T).view(S, self.idx_h, self.idx_d)
        q = torch.cat([q[..., : self.idx_d - self.rd],
                       rot_pairs(q[..., self.idx_d - self.rd:], cq, sq)], -1)
        w = (x @ L["attn.indexer.weights_proj.weight"].to(x.dtype).T).float() \
            * (self.idx_d ** -0.5) * (self.idx_h ** -0.5)             # (S, idx_h)
        ent = self.compress(L, x, ratio, overlapping=True, indexer=True)   # (T, idx_d)
        scores = F.relu(torch.einsum("shd,td->sht", q.float(), ent.float()))
        scores = (scores * w[..., None]).sum(1)                      # (S, T) over heads
        k = min(self.topk_idx, T)
        if T == 0 or k == 0:
            return torch.zeros((S, 0), dtype=torch.long, device=self.dev)
        bound = ((pos + 1) // ratio)[:, None]
        scores = scores.masked_fill(torch.arange(T, device=self.dev)[None, :] >= bound,
                                    float("-inf"))
        sel = scores.topk(k, -1).indices
        return torch.where(sel >= bound, torch.full_like(sel, -1), sel + offset)

    # ---------------------------------------------------------------- MoE
    def swiglu(self, x, w1, w2, w3, li=None, prefix=None):
        """li/prefix given ONLY for the shared expert (LoRA sites); routed experts
        call plain (no adapters on routed experts, per dummy/official target set)."""
        lim = self.swiglu_limit
        if prefix is None:
            gate = (x @ w1.to(x.dtype).T).float().clamp(max=lim)
            up = (x @ w3.to(x.dtype).T).float().clamp(min=-lim, max=lim)
            return ((F.silu(gate) * up).to(self.act_dtype)
                    @ w2.to(self.act_dtype).T).float()
        gate = self._lmm(x, w1, li, prefix + ".w1.weight").float().clamp(max=lim)
        up = self._lmm(x, w3, li, prefix + ".w3.weight").float().clamp(min=-lim, max=lim)
        h = (F.silu(gate) * up).to(self.act_dtype)
        return self._lmm(h, w2, li, prefix + ".w2.weight").float()

    def ffn(self, i, L, x, ids):
        """Official MoE.forward + Gate. Route FIRST, dequant only routed experts."""
        xf = x.float()
        logits = xf @ L["ffn.gate.weight"].float().T
        scores = F.softplus(logits).sqrt()
        if i < self.hash_layers:
            sel = L["ffn.gate.tid2eid"][ids]                         # (S, k)
        else:
            sel = (scores + L["ffn.gate.bias"]).topk(self.exp_k, -1).indices
        w = scores.gather(1, sel)
        w = w / (w.sum(-1, keepdim=True)) * self.route_scale
        y = self.swiglu(x, L["ffn.shared_experts.w1.weight"],
                        L["ffn.shared_experts.w2.weight"],
                        L["ffn.shared_experts.w3.weight"],
                        li=i, prefix="ffn.shared_experts")           # shared expert
        for e in sel.flatten().unique():
            w_e = (w * (sel == e)).sum(-1)                           # per-token weight
            rows = w_e.nonzero().flatten()
            if rows.numel() == 0:
                continue
            w1, w2, w3 = self.L.load_expert(i, int(e))
            y[rows] += w_e[rows, None] * self.swiglu(x[rows], w1, w2, w3)
            del w1, w2, w3
        return y.to(self.act_dtype)

    # ---------------------------------------------------------------- hyper-connections
    def hc_pre(self, st, fn, base, scale):
        """Official Block.hc_pre: mixes = linear(flat)*rsqrt; sinkhorn split;
        collapse over streams. st (1,S,hc,H) bf16 -> collapsed (1,S,H) bf16."""
        shape = st.shape
        x = st.flatten(2).float()
        rs = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        mixes = (x @ fn.float().T) * rs                             # (1,S,24)
        pre = torch.sigmoid(mixes[..., : self.hc] * scale[0] + base[: self.hc]) + self.hc_eps
        post = 2.0 * torch.sigmoid(
            mixes[..., self.hc: 2 * self.hc] * scale[1] + base[self.hc: 2 * self.hc])
        comb = mixes[..., 2 * self.hc:].view(1, -1, self.hc, self.hc) * scale[2] \
            + base[2 * self.hc:].view(self.hc, self.hc)
        comb = torch.softmax(comb, -1) + self.hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        y = (pre.unsqueeze(-1) * x.view(shape)).sum(2)
        return y.to(st.dtype), post, comb

    def hc_post(self, y, st, post, comb):
        """Official Block.hc_post: st <- post*y + sum_j comb[...,i,j]*st[...,j,:]
        (UNtransposed, official). HC_COMB_TRANSPOSED=1 swaps to exl3 orientation."""
        m = comb.transpose(-1, -2) if HC_COMB_TRANSPOSED else comb
        out = post.unsqueeze(-1) * y.unsqueeze(0).unsqueeze(2).float() \
            + (m.unsqueeze(-1) * st.float().unsqueeze(-2)).sum(2)
        return out.to(y.dtype)

    def hc_head(self, st):
        x = st.flatten(2).float()
        rs = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        mixes = (x @ self.top["hc_head_fn"].float().T) * rs
        pre = torch.sigmoid(mixes * self.top["hc_head_scale"] + self.top["hc_head_base"]) \
            + self.hc_eps
        return (pre.unsqueeze(-1) * x.view(st.shape)).sum(2).to(st.dtype)

    # ---------------------------------------------------------------- block + model
    def block(self, i, L, st, ids):
        x, post, comb = self.hc_pre(st, L["hc_attn_fn"], L["hc_attn_base"], L["hc_attn_scale"])
        xn = rms_w(x[0], L["attn_norm.weight"], self.eps)
        y = self.attention(i, L, xn)
        st = self.hc_post(y, st, post, comb)

        x, post, comb = self.hc_pre(st, L["hc_ffn_fn"], L["hc_ffn_base"], L["hc_ffn_scale"])
        xn = rms_w(x[0], L["ffn_norm.weight"], self.eps)
        y = self.ffn(i, L, xn, ids)
        st = self.hc_post(y, st, post, comb)
        return st

    def forward(self, input_ids, layer_cb=None):
        """input_ids: (S,) long tensor on device. Returns logits (S, vocab) fp32.
        Streams ALL layers one at a time (peak resident: 1 layer + top stack)."""
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.dev)
        S = ids.shape[0]
        h = self.top["embed"][ids].to(self.act_dtype)
        st = h.unsqueeze(0).unsqueeze(2).expand(1, S, self.hc, self.H).contiguous()
        for i in range(self.n_layers):
            t0 = time.time()
            L = self._load_layer(i)
            st = self.block(i, L, st, ids)
            del L
            if layer_cb is not None:
                layer_cb(i, st, time.time() - t0)
        h = self.hc_head(st)[0]                                      # (S, H) bf16
        h = rms_w(h, self.top["norm"], self.eps)
        return h.float() @ self.top["head_f32"].T

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=20):
        ids = [int(t) for t in input_ids]
        gen = []
        for _ in range(max_new_tokens):
            logits = self.forward(ids)
            nxt = int(logits[-1].argmax())
            ids.append(nxt)
            gen.append(nxt)
            if nxt == self.eos:
                break
        return ids, gen

    # ------------------------------------------------------------- batched path
    # Packing-based batch forward: N variable-length sequences concatenated into one
    # token stream; per-token projections/hc/MoE are pack-agnostic; attention builds
    # per-token gather rows (window ++ own-sequence pool) so sequences never interact.
    # Same math as the single-sequence path (validated by smoke G1-G3); used by
    # tc_batch_gen.py for on-policy data regen.

    def _attn_batch(self, i, L, x, seq_of, pos_of, starts, lens):
        S = x.shape[0]
        N = len(starts)
        ratio = self.ratios[i]
        invf = self._invf(ratio)
        coss, sins = rope_cs(invf, pos_of)                           # per-token pos
        cq, sq = coss.view(S, 1, self.rd // 2), sins.view(S, 1, self.rd // 2)

        qr = rms_w(x @ L["attn.wq_a.weight"].to(x.dtype).T, L["attn.q_norm.weight"], self.eps)
        q = self._lmm(qr, L["attn.wq_b.weight"], i,
                       "attn.wq_b.weight").view(S, self.n_heads, self.hd)
        q = rms_u(q, self.eps)
        q = torch.cat([q[..., : self.nope], rot_pairs(q[..., self.nope:], cq, sq)], -1)
        kv = rms_w(self._lmm(x, L["attn.wkv.weight"], i, "attn.wkv.weight"),
                   L["attn.kv_norm.weight"], self.eps)
        kv = torch.cat([kv[:, : self.nope], rot_pairs(kv[:, self.nope:], coss, sins)], -1)

        # window rows: candidates in the PACK (own sequence only, causal, w back)
        st_t = torch.tensor(starts, device=self.dev, dtype=torch.long)[seq_of]
        j = torch.arange(self.window, device=self.dev)
        cand = (st_t + (pos_of - self.window + 1).clamp(min=0)).unsqueeze(1) + j[None, :]
        cand = torch.where(j[None, :] > pos_of[:, None],
                           torch.full_like(cand, -1), cand)          # (S, W)

        parts = [cand]
        if ratio:
            pool = []
            for k in range(N):
                s0, ln = starts[k], lens[k]
                ent_k = self.compress(L, x[s0:s0 + ln], ratio,
                                      overlapping=(ratio == 4), indexer=False)
                pool.append(ent_k)
            # global pool offsets (into kv_all after the S packed window tokens)
            offs, acc = [], 0
            for p in pool:
                offs.append(S + acc)
                acc += p.shape[0]
            pool_all = torch.cat(pool, 0) if acc else torch.zeros(
                (0, self.hd), dtype=torch.bfloat16, device=self.dev)
            rows = []
            K_p = 0
            for k in range(N):
                s0, ln = starts[k], lens[k]
                ent_k = pool[k]
                T_k = ent_k.shape[0]
                if ratio == 4:  # CSA: indexer selection per sequence
                    sel_k = self.indexer_topk(i, L, x[s0:s0 + ln], qr[s0:s0 + ln],
                                              T_k, offset=offs[k])   # (ln, k_k)
                else:           # HCA: dense causal bound
                    t_loc = torch.arange(ln, device=self.dev)
                    mat = torch.arange(T_k, device=self.dev)[None, :].repeat(ln, 1)
                    bound = ((t_loc + 1) // ratio)[:, None]
                    sel_k = torch.where(mat >= bound, torch.full_like(mat, -1),
                                        mat + offs[k])
                K_p = max(K_p, sel_k.shape[1])
                rows.append(sel_k)
            pad = torch.full((S, K_p), -1, dtype=torch.long, device=self.dev)
            for k in range(N):
                s0, ln = starts[k], lens[k]
                pad[s0:s0 + ln, :rows[k].shape[1]] = rows[k]
            parts.append(pad)
            kv_all = torch.cat([kv, pool_all], 0)
        else:
            kv_all = kv
        idx = torch.cat(parts, -1)                                   # (S, W + K_p)

        o = self.sparse_attn(q, kv_all, L["attn.attn_sink"], idx)
        o = torch.cat([o[..., : self.nope], rot_pairs(o[..., self.nope:], cq, -sq)], -1)
        og = o.reshape(S, self.og, self.hpg * self.hd)
        woa = L["attn.wo_a.weight"].to(og.dtype).view(self.og, self.olr, self.hpg * self.hd)
        y = torch.einsum("sgd,grd->sgr", og, woa).reshape(S, self.og * self.olr)
        return self._lmm(y, L["attn.wo_b.weight"], i, "attn.wo_b.weight")

    def block_batch(self, i, L, st, ids, seq_of, pos_of, starts, lens):
        x, post, comb = self.hc_pre(st, L["hc_attn_fn"], L["hc_attn_base"], L["hc_attn_scale"])
        xn = rms_w(x[0], L["attn_norm.weight"], self.eps)
        y = self._attn_batch(i, L, xn, seq_of, pos_of, starts, lens)
        st = self.hc_post(y, st, post, comb)

        x, post, comb = self.hc_pre(st, L["hc_ffn_fn"], L["hc_ffn_base"], L["hc_ffn_scale"])
        xn = rms_w(x[0], L["ffn_norm.weight"], self.eps)
        y = self.ffn(i, L, xn, ids)
        st = self.hc_post(y, st, post, comb)
        return st

    @torch.inference_mode()
    def forward_batch(self, seqs, layer_cb=None):
        """seqs: list of N variable-length id lists. Returns (N, vocab) fp32 logits
        at each sequence's LAST token."""
        starts, lens, flat = [], [], []
        for s in seqs:
            starts.append(len(flat))
            lens.append(len(s))
            flat.extend(int(t) for t in s)
        ids = torch.tensor(flat, dtype=torch.long, device=self.dev)
        S = ids.shape[0]
        seq_of = torch.repeat_interleave(
            torch.arange(len(seqs), device=self.dev),
            torch.tensor(lens, device=self.dev))
        pos_of = torch.arange(S, device=self.dev) - \
            torch.tensor(starts, device=self.dev, dtype=torch.long)[seq_of]
        h = self.top["embed"][ids].to(self.act_dtype)
        st = h.unsqueeze(0).unsqueeze(2).expand(1, S, self.hc, self.H).contiguous()
        for i in range(self.n_layers):
            t0 = time.time()
            L = self._load_layer(i)
            st = self.block_batch(i, L, st, ids, seq_of, pos_of, starts, lens)
            del L
            if layer_cb is not None:
                layer_cb(i, st, time.time() - t0)
        h = self.hc_head(st)[0]
        h = rms_w(h, self.top["norm"], self.eps)
        last = torch.tensor([s + l - 1 for s, l in zip(starts, lens)],
                            device=self.dev)
        return h[last].float() @ self.top["head_f32"].T
