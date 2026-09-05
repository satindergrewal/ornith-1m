#!/usr/bin/env python3
"""glm_full.py — full-model GLM-5.3-Flash engine on the real FP8 abliterated
checkpoint, layer-streamed (mirrors dsv4_full.py; the DSV4 harness is the
structural template, GLM semantics from the fork's own code).

Math authorities (every line cites its source):
  * KDA layers: fork vllm/models/glm5next/nvidia/kda.py + the verified dummy
    engine nano_torch_glm.py (same math, degenerate width). The delta-rule
    recurrence runs through upstream fla 0.5.2 fla.ops.kda.chunk_kda
    (autograd-complete, training-capable). The log-decay gate is computed in
    torch exactly per fla.ops.kda.gate.naive_kda_lowerbound_gate:
        g = -exp(A_log) * sigmoid(clamp(g1 + dt_bias, min=-5))
    which bit-matches the dummy's decay = exp(-exp(A_log)*gate). Parity of
    chunk_kda vs the per-token loop is a smoke gate (G0), not an assumption.
  * DSA layers (3,7,...,43): fork glm5next/nvidia/attention.py
    Glm5NextMLAAttention with qk_rope_head_dim=0 (mla_use_nope) — NO rope
    anywhere. At ThinkingCap sequence lengths (S <= 2048 < index_topk), the
    indexer top-2048 selects every causal token, so sparse MLA reduces to
    DENSE causal MLA — exact, documented (same class of simplification as
    the DSV4 engine's smoke note). The indexer is therefore not loaded.
  * MoE: fork model.py Glm5NextMoE — sigmoid scoring, noaux_tc
    e_score_correction_bias (selection only), weights from unbiased scores,
    renormalized, x routed_scaling 2.5, SwiGLU clamp 10.
  * hc: mhc v2 (hc_pre/hc_post identical to dsv4_full.py; final contract =
    MEAN over streams + final norm — no hc head tensor in this checkpoint).

Env knobs: none (GLM_MODEL / GLM_META drive glm_full_loader.StreamingGLM).
"""

import math
import os
import time

import torch
import torch.nn.functional as F

from glm_full_loader import StreamingGLM


# ----------------------------------------------------------------- primitives

def rms_w(x, w, eps):
    d = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (w.float() * xf).to(d)


def causal_conv_silu(x, w, k):
    """Merged causal depthwise conv + silu over (S, C); w (C, k) fp32.
    y[t, c] = sum_j w[c, j] x[t-k+1+j, c] (zero left pad)."""
    S, C = x.shape
    xp = F.pad(x.T.unsqueeze(0).float(), (k - 1, 0))
    y = F.conv1d(xp, w.unsqueeze(1), groups=C)
    return F.silu(y[0].T)


# ----------------------------------------------------------------- engine

class GLMFull:
    def __init__(self, loader: StreamingGLM, load_top=True):
        self.L = loader
        c = loader.cfg
        self.cfg = c
        self.dev = loader.device
        self.n_layers = c["num_hidden_layers"]
        self.H = c["hidden_size"]
        self.eps = c["rms_norm_eps"]
        self.hc = c["hc_mult"]
        self.hc_iters, self.hc_eps = c["hc_sinkhorn_iters"], c["hc_eps"]
        self.eos = c.get("eos_token_id", [])
        self.eos = self.eos if isinstance(self.eos, list) else [self.eos]

        # KDA (linear attention) config
        la = c.get("linear_attn_config") or {}
        self.kda_heads = la.get("num_heads", c.get("linear_num_heads", 64))
        self.kda_hd = la.get("head_dim", c.get("linear_head_dim", 128))
        self.conv_k = la.get("short_conv_kernel_size",
                             c.get("linear_conv_kernel_dim", 4))
        self.lower_bound = la.get("gate_lower_bound",
                                  c.get("linear_lower_bound", -5.0))
        self.kda_proj = self.kda_heads * self.kda_hd              # 8192

        # DSA (MLA) config — nope-only (qk_rope_head_dim = 0)
        self.n_heads = c["num_attention_heads"]                   # 64
        self.qk_nope = c["qk_nope_head_dim"]                      # 256
        self.qk_rope = c.get("qk_rope_head_dim", 0)               # 0
        self.qk = self.qk_nope + self.qk_rope                     # 256
        self.v_hd = c["v_head_dim"]                               # 256
        self.qlr = c["q_lora_rank"]                               # 1536
        self.kvlr = c["kv_lora_rank"]                             # 512
        self.mla_scale = self.qk ** -0.5
        assert self.qk_rope == 0, "rope dims present; engine assumes nope-only"

        # layer types
        lt = c.get("layer_types")
        self.layer_types = lt if lt else [
            "deepseek_sparse_attention" if (i % 4 == 3 and i >= 3)
            else "linear_attention" for i in range(self.n_layers)]
        self.dsa = {i for i, t in enumerate(self.layer_types)
                    if t != "linear_attention"}

        # MoE config
        self.n_exp, self.exp_k = c["n_routed_experts"], c["num_experts_per_tok"]
        self.route_scale = c["routed_scaling_factor"]
        self.swiglu_limit = c["swiglu_limit"]
        self.first_dense = c.get("first_k_dense_replace", 0)

        self.top = loader.load_top() if load_top else None
        if self.top is not None:
            self.top["head_f32"] = self.top["head"].float()

        self.act_dtype = torch.bfloat16
        # LoRA state (same protocol as dsv4_full.attach_lora)
        self.lora = None
        self.lora_scale = 1.0
        self.lora_fold = False

    # ---------------------------------------------------------------- helpers
    def attach_lora(self, path, scale, fold=False):
        from safetensors.torch import load_file
        t = load_file(path)
        self.lora = {}
        self.lora_scale = scale
        self.lora_fold = fold
        for k in t:
            if not k.startswith("lora.") or not (k.endswith(".A") or k.endswith(".B")):
                continue
            site = k[len("lora."):-2]
            li = int(site.split(".")[1])
            key = (li, site)
            d = self.lora.setdefault(key, {})
            d[k[-1]] = t[k].to(self.dev).float()
        self.lora = {k: (v["A"], v["B"]) for k, v in self.lora.items()
                     if "A" in v and "B" in v}
        return len(self.lora)

    def _lmm(self, x, W, i, site):
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

    # ---------------------------------------------------------------- KDA layer
    def kda_attn_batch(self, i, L, x, cu_seqlens):
        """x (S, H) bf16 normed -> (S, H) bf16. Row-isolated via cu_seqlens."""
        from fla.ops.kda import chunk_kda
        S = x.shape[0]
        h, d = self.kda_heads, self.kda_hd
        q = (x @ L["self_attn.q_proj.weight"].to(x.dtype).T).view(S, h, d)
        k = (x @ L["self_attn.k_proj.weight"].to(x.dtype).T).view(S, h, d)
        v = (x @ L["self_attn.v_proj.weight"].to(x.dtype).T).view(S, h, d)

        conv_w = torch.cat([
            L["self_attn.q_conv1d.weight"].reshape(self.kda_proj, -1).float(),
            L["self_attn.k_conv1d.weight"].reshape(self.kda_proj, -1).float(),
            L["self_attn.v_conv1d.weight"].reshape(self.kda_proj, -1).float()], 0)
        qkv = causal_conv_silu(
            torch.cat([q.reshape(S, -1), k.reshape(S, -1), v.reshape(S, -1)], -1),
            conv_w, self.conv_k)
        q, k, v = qkv.split(self.kda_proj, dim=-1)
        q = q.reshape(S, h, d).float()
        k = k.reshape(S, h, d).float()
        v = v.reshape(S, h, d).float()

        q = q / (q.norm(dim=-1, keepdim=True) + 1e-20)      # use_qk_l2norm
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-20)

        beta = torch.sigmoid(
            (x @ L["self_attn.b_proj.weight"].to(x.dtype).T).float())[:, :h]

        g1 = ((x @ L["self_attn.f_a_proj.weight"].to(x.dtype).T).float()
              @ L["self_attn.f_b_proj.weight"].float().T).view(S, h, d)
        g2 = ((x @ L["self_attn.g_a_proj.weight"].to(x.dtype).T).float()
              @ L["self_attn.g_b_proj.weight"].float().T).view(S, h, d)
        dt_bias = L["self_attn.dt_bias"].float().view(h, d)
        A_log = L["self_attn.A_log"].float().view(1, h, 1)
        # log-decay gate, per fla gate.naive_kda_lowerbound_gate == the dummy
        gate = (g1 + dt_bias).clamp(min=self.lower_bound).sigmoid()
        g = -A_log.exp() * gate                              # (S, h, d) <= 0

        B = 1
        o, _ = chunk_kda(
            q.unsqueeze(0).to(self.act_dtype), k.unsqueeze(0).to(self.act_dtype),
            v.unsqueeze(0).to(self.act_dtype), g.unsqueeze(0),
            beta.unsqueeze(0),
            scale=d ** -0.5,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False, safe_gate=False,
            cu_seqlens=cu_seqlens,
        )
        o = o.squeeze(0).float()                             # (S, h, d)
        normed = rms_w(o.reshape(S, h, d), L["self_attn.o_norm.weight"], self.eps)
        core = (normed * g2.sigmoid()).reshape(S, self.kda_proj)
        return self._lmm(core.to(self.act_dtype), L["self_attn.o_proj.weight"],
                         i, "self_attn.o_proj.weight")

    # ---------------------------------------------------------------- DSA layer
    def mla_attn_batch(self, i, L, x, seq_of, pos_of, starts, lens, q_block=512):
        """Dense causal MLA over the packed rows (S <= index_topk: the indexer
        would select every causal token). q-blocked to bound the (S,S,h)
        logits materialization in training windows."""
        S = x.shape[0]
        h, dk, dv = self.n_heads, self.qk, self.v_hd
        qr = rms_w(self._lmm(x, L["self_attn.q_a_proj.weight"], i,
                             "self_attn.q_a_proj.weight").float(),
                   L["self_attn.q_a_layernorm.weight"].float(), self.eps).to(x.dtype)
        q = self._lmm(qr, L["self_attn.q_b_proj.weight"], i,
                      "self_attn.q_b_proj.weight").view(S, h, dk)
        kv = rms_w(self._lmm(x, L["self_attn.kv_a_proj_with_mqa.weight"], i,
                             "self_attn.kv_a_proj_with_mqa.weight").float(),
                   L["self_attn.kv_a_layernorm.weight"].float(), self.eps).to(x.dtype)
        kvb = self._lmm(kv, L["self_attn.kv_b_proj.weight"], i,
                        "self_attn.kv_b_proj.weight").view(S, h, dk + dv)
        kk = kvb[..., :dk].float()
        vv = kvb[..., dk:].float()

        pos = pos_of
        causal = (seq_of[None, :] == seq_of[:, None]) & \
                 (pos[None, :] <= pos[:, None])               # (S_key, S_q)
        out = torch.empty(S, h, dv, dtype=torch.float32, device=self.dev)
        for s0 in range(0, S, q_block):
            s1 = min(s0 + q_block, S)
            logits = torch.einsum("qhd,khd->qhk", q[s0:s1].float(), kk) \
                * self.mla_scale                             # (q, h, k)
            logits = logits.masked_fill(~causal[:, s0:s1].T.unsqueeze(1),
                                        float("-inf"))        # (q, 1, k)
            p = torch.softmax(logits, -1)
            out[s0:s1] = torch.einsum("qhk,khd->qhd", p, vv)
        y = out.reshape(S, h * dv).to(self.act_dtype)
        return self._lmm(y, L["self_attn.o_proj.weight"], i,
                         "self_attn.o_proj.weight")

    # ---------------------------------------------------------------- attention
    def attn_batch(self, i, L, x, ids, seq_of, pos_of, starts, lens, cu_seqlens):
        zero = os.environ.get("GLM_ZERO_ATTN", "")    # "dsa"/"kda" ablations
        if i in self.dsa and zero == "dsa":
            return torch.zeros_like(x)
        if i not in self.dsa and zero == "kda":
            return torch.zeros_like(x)
        if i in self.dsa:
            return self.mla_attn_batch(i, L, x, seq_of, pos_of, starts, lens)
        return self.kda_attn_batch(i, L, x, cu_seqlens)

    # ---------------------------------------------------------------- MoE / FFN
    def swiglu(self, x, wg, wu, wd, li=None, prefix=None):
        lim = self.swiglu_limit
        if prefix is None:
            g = (x @ wg.to(x.dtype).T).float().clamp(max=lim)
            u = (x @ wu.to(x.dtype).T).float().clamp(min=-lim, max=lim)
            return ((F.silu(g) * u).to(self.act_dtype)
                    @ wd.to(self.act_dtype).T).float()
        g = self._lmm(x, wg, li, prefix + ".weight").float().clamp(max=lim)
        u = self._lmm(x, wu, li, prefix + ".weight").float().clamp(
            min=-lim, max=lim)
        hact = (F.silu(g) * u).to(self.act_dtype)
        return self._lmm(hact, wd, li, prefix + ".weight").float()

    def ffn(self, i, L, x, ids):
        if "mlp.gate.weight" not in L:                       # dense L0..first_dense
            return self._dense_ffn(i, L, x)
        if os.environ.get("GLM_ZERO_ROUTED") == "1":         # ablation: shared only
            return self.swiglu(x, L["mlp.shared_experts.gate_proj.weight"],
                               L["mlp.shared_experts.up_proj.weight"],
                               L["mlp.shared_experts.down_proj.weight"],
                               li=i, prefix="mlp.shared_experts.gate_proj"
                               ).to(self.act_dtype)
        xf = x.float()
        logits = xf @ L["mlp.gate.weight"].float().T
        scores = logits.sigmoid()                            # sigmoid scoring
        sel = (scores + L["mlp.gate.e_score_correction_bias"].float()) \
            .topk(self.exp_k, -1).indices                    # noaux_tc: bias selects
        w = scores.gather(1, sel)
        w = w / (w.sum(-1, keepdim=True)) * self.route_scale
        y = self.swiglu(x, L["mlp.shared_experts.gate_proj.weight"],
                        L["mlp.shared_experts.up_proj.weight"],
                        L["mlp.shared_experts.down_proj.weight"],
                        li=i, prefix="mlp.shared_experts.gate_proj")
        for e in sel.flatten().unique():
            w_e = (w * (sel == e)).sum(-1)
            rows = w_e.nonzero().flatten()
            if rows.numel() == 0:
                continue
            wg, wu, wd = self.L.load_expert(i, int(e))
            y[rows] += w_e[rows, None] * self.swiglu(x[rows], wg, wu, wd)
            del wg, wu, wd
        return y.to(self.act_dtype)

    def _dense_ffn(self, i, L, x):
        g = self._lmm(x, L["mlp.gate_proj.weight"], i,
                      "mlp.gate_proj.weight").float().clamp(max=self.swiglu_limit)
        u = self._lmm(x, L["mlp.up_proj.weight"], i,
                      "mlp.up_proj.weight").float().clamp(
            min=-self.swiglu_limit, max=self.swiglu_limit)
        hact = (F.silu(g) * u).to(self.act_dtype)
        return self._lmm(hact, L["mlp.down_proj.weight"], i,
                         "mlp.down_proj.weight").to(self.act_dtype)

    # ---------------------------------------------------------------- hc
    def hc_pre(self, st, fn, base, scale):
        shape = st.shape
        x = st.flatten(2).float()
        rs = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        mixes = (x @ fn.float().T) * rs
        pre = torch.sigmoid(mixes[..., : self.hc] * scale[0] + base[: self.hc]) \
            + self.hc_eps
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
        """fork mhc_post_torch: out_j = post_j*y + sum_i comb[i,j]*st_i —
        the comb-TRANSPOSED orientation. GLM_HC_POST_TRANSPOSED=0 flips to
        the DSV4 orientation (out_i = sum_j comb[i,j]*st_j) for A/B tests."""
        if os.environ.get("GLM_HC_POST_TRANSPOSED", "1") == "1":
            mixed = torch.einsum("bsij,bsih->bsjh", comb.float(), st.float())
        else:
            mixed = torch.einsum("bsij,bsjh->bsih", comb.float(), st.float())
        out = post.unsqueeze(-1) * y.unsqueeze(0).unsqueeze(2).float() + mixed
        return out.to(y.dtype)

    def hc_head(self, st):
        """Final hc contract = MEAN over streams (no hc head tensor here)."""
        return st.float().mean(2).to(st.dtype)               # (1, S, H)

    # ---------------------------------------------------------------- block
    def block_batch(self, i, L, st, ids, seq_of, pos_of, starts, lens, cu_seqlens):
        x, post, comb = self.hc_pre(st, L["hc_attn_fn"], L["hc_attn_base"],
                                    L["hc_attn_scale"])
        xn = rms_w(x[0], L["input_layernorm.weight"], self.eps)
        y = self.attn_batch(i, L, xn, ids, seq_of, pos_of, starts, lens, cu_seqlens)
        st = self.hc_post(y, st, post, comb)

        x, post, comb = self.hc_pre(st, L["hc_ffn_fn"], L["hc_ffn_base"],
                                    L["hc_ffn_scale"])
        xn = rms_w(x[0], L["post_attention_layernorm.weight"], self.eps)
        y = self.ffn(i, L, xn, ids)
        st = self.hc_post(y, st, post, comb)
        return st

    # ---------------------------------------------------------------- model
    @torch.inference_mode()
    def forward_batch(self, seqs, layer_cb=None):
        """seqs: list of N variable-length id lists -> (N, vocab) fp32 logits
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
        cu = [0] + list(torch.tensor(lens).cumsum(0).tolist())
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=self.dev)

        h = self.top["embed"][ids].to(self.act_dtype)
        st = h.unsqueeze(0).unsqueeze(2).expand(1, S, self.hc, self.H).contiguous()
        for i in range(self.n_layers):
            t0 = time.time()
            L = self._load_layer(i)
            st = self.block_batch(i, L, st, ids, seq_of, pos_of, starts, lens,
                                  cu_seqlens)
            del L
            if layer_cb is not None:
                layer_cb(i, st, time.time() - t0)
        h = self.hc_head(st)[0]                               # (S, H)
        h = rms_w(h, self.top["norm"], self.eps)
        last = torch.tensor([s + l - 1 for s, l in zip(starts, lens)],
                            device=self.dev)
        return h[last].float() @ self.top["head_f32"].T

    def forward(self, input_ids, layer_cb=None):
        """Single sequence -> LAST-TOKEN logits (vocab,) fp32 (the
        forward_batch contract, unwrapped)."""
        return self.forward_batch([list(int(t) for t in input_ids)],
                                  layer_cb)[0]

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=20):
        ids = [int(t) for t in input_ids]
        gen = []
        for _ in range(max_new_tokens):
            nxt = int(self.forward(ids).argmax())
            ids.append(nxt)
            gen.append(nxt)
            if nxt in self.eos:
                break
        return ids, gen
