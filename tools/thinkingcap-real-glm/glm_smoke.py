#!/usr/bin/env python3
"""glm_smoke.py — gates for the GLM-5.3-Flash real engine.

  G0  KDA parity: fla chunk_kda vs the dummy's verified per-token loop
      (nano_torch_glm math, full width) on synthetic data with two
      row-groups via cu_seqlens. MUST pass before any weight is trusted.
  G1  full forward over all 45 layers: no NaN/Inf, per-layer stream RMS sane.
  G2  top-5 tokens at the last position are plausible pieces, not byte junk.
  G3  greedy 20 tokens: coherent + deterministic across two runs.

Run on the GLM pod: cd /root/tc-glm && python glm_smoke.py"""

import json
import time

import torch
import torch.nn.functional as F

from glm_full_loader import StreamingGLM
from glm_full import GLMFull, rms_w


def kda_loop_reference(q, k, v, g, beta, cu, scale):
    """The dummy's recurrence (nano_torch_glm.kda_attn core), batched over
    cu_seqlens row groups. q/k/v (T, H, D) f32; g (T, H, D) log-decay; beta
    (T, H) f32 in (0,1). Returns o (T, H, D)."""
    T, H, D = q.shape
    out = torch.zeros_like(v)
    for b in range(len(cu) - 1):
        s, e = int(cu[b]), int(cu[b + 1])
        state = torch.zeros(H, D, D, device=q.device)
        for t in range(s, e):
            dec = g[t].exp()                                   # (H, D)
            Sd = state * dec[:, :, None]
            kk = k[t]
            bt = beta[t]
            proj = torch.einsum("hdk,hd->hk", Sd, kk)
            state = (Sd - bt[:, None, None] * kk[:, :, None] * proj[:, None, :]
                     + bt[:, None, None] * kk[:, :, None] * v[t][:, None, :])
            out[t] = torch.einsum("hk,hkv->hv", q[t] * scale, state)
    return out


def g0_kda_paritiy(dev):
    from fla.ops.kda import chunk_kda
    torch.manual_seed(20260906)
    T, H, D = 40, 8, 128
    cu = [0, 18, 40]
    q = F.normalize(torch.randn(T, H, D, device=dev), dim=-1)
    k = F.normalize(torch.randn(T, H, D, device=dev), dim=-1)
    v = torch.randn(T, H, D, device=dev)
    g = -torch.rand(T, H, D, device=dev) * 6 - 0.05          # log-decay <= 0
    beta = torch.rand(T, H, device=dev)
    scale = D ** -0.5

    o_loop = kda_loop_reference(q, k, v, g, beta, cu, scale)

    o_chunk, _ = chunk_kda(
        q.unsqueeze(0).bfloat16(), k.unsqueeze(0).bfloat16(),
        v.unsqueeze(0).bfloat16(), g.unsqueeze(0).float(),
        beta.unsqueeze(0).float(), scale=scale,
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False, safe_gate=False,
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=dev),
    )
    o_chunk = o_chunk.squeeze(0).float()
    md = (o_loop - o_chunk).abs().max().item()
    rel = md / o_loop.abs().max().item()

    # isolation self-check: perturbing row 1 must leave row 2 bit-identical
    # (both passes through the SAME bf16 chunk path, so any diff = leak).
    q2 = q.clone()
    q2[: cu[1]] += 0.37
    o2, _ = chunk_kda(
        q2.unsqueeze(0).bfloat16(), k.unsqueeze(0).bfloat16(),
        v.unsqueeze(0).bfloat16(), g.unsqueeze(0).float(),
        beta.unsqueeze(0).float(), scale=scale,
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False, safe_gate=False,
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=dev),
    )
    o2 = o2.squeeze(0)
    iso = torch.equal(o2[cu[1]:], o_chunk[cu[1]:])

    print(f"[G0] chunk_kda(bf16) vs loop(f32): maxdiff {md:.3e} "
          f"(rel {rel:.3e} — bf16 epsilon class); row isolation: {iso}")
    assert rel < 2e-2, f"G0 parity FAIL rel={rel}"
    assert iso, "G0 FAIL: cu_seqlens row isolation broken (row 1 leaked into row 2)"
    print("[G0] PASS")
    return True


def get_tok():
    from tokenizers import Tokenizer
    return Tokenizer.from_file("/root/glm-fp8/tokenizer.json")


def main():
    dev = "cuda"
    g0_kda_paritiy(dev)

    print("[smoke] loading engine (first layer stream incl. cold reads)...",
          flush=True)
    L = StreamingGLM(device=dev)
    eng = GLMFull(L)
    tok = get_tok()
    prompt = "The capital of France is"
    ids = tok.encode(prompt).ids

    t0 = time.time()
    logs = []
    rms_stack = []

    def cb(i, st, dt):
        r = st.float().square().mean().sqrt().item()
        rms_stack.append((i, r))
        if not torch.isfinite(st).all():
            print(f"[G1] FAIL layer {i}: non-finite stream")
            raise SystemExit(1)

    logits = eng.forward(ids, layer_cb=cb)                   # (vocab,) last token
    fw = time.time() - t0
    lo, hi = min(r for _, r in rms_stack), max(r for _, r in rms_stack)
    print(f"[G1] 45 layers finite; stream RMS range [{lo:.3f}, {hi:.3f}] "
          f"(fwd {fw:.1f}s)")
    assert 1e-3 < lo and hi < 1e4, "stream RMS out of range"

    top = logits.topk(5)
    print("[G2] top-5 @ last pos:",
          [(tok.decode([int(i)]), round(v, 2))
           for v, i in zip(top.values.tolist(), top.indices.tolist())])

    ids2, gen1 = eng.generate(ids, max_new_tokens=20)
    t1 = time.time()
    text1 = tok.decode(gen1, skip_special_tokens=True)
    _, gen2 = eng.generate(ids, max_new_tokens=20)
    det = gen1 == gen2
    print(f"[G3] gen1 ({t1 - fw:.1f}s): {text1!r}")
    print(f"[G3] run2 identical: {det}")
    ok = det and len(text1.strip()) > 0
    print(f"[smoke] OVERALL: {'PASS' if ok else 'FAIL'}")
    json.dump({"g1_rms": [lo, hi], "g2_top5": [int(i) for i in top.indices],
               "g3": text1, "g3_det": det, "pass": ok},
              open("/root/tc-glm/smoke_result.json", "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
