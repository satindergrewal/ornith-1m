# Ablation fallback: if the first cap underperforms

Trigger (exact): the full-site cap at the sweep's best viable alpha FAILS
the holdout gate: accuracy delta < -2/36 vs base, OR every alpha passes
accuracy but shows no length reduction. In the no-length-reduction case,
check the training loss curve first (flat loss = the cap never learned
termination; the arms below will not fix that).

## Arms (one variable: the site set)

| arm | sites per language layer 0..42 | adapters | what it holds |
|-----|--------------------------------|----------|---------------|
| A   | attn wq_b, wkv, wo_b + shared_experts w1, w2, w3 | 258 | the failed baseline; DO NOT rerun, reuse its training log |
| B   | attn wq_b, wkv, wo_b | 129 | attention-only |
| C   | shared_experts w1, w2, w3 | 129 | shared-experts-only |

Everything else frozen at the failed A run's values: same SFT rows, train
seed, R/alpha/LR/epochs, same holdout battery and paired sample seed. Exact
adapter and param counts print at trainer startup (shapes are read from the
checkpoint meta); B and C are each half of A by site count. Routed experts,
hc params, norms, sinks, indexer, MTP/DSpark remain untouched in every arm,
same reasons as the 04-real target memo.

## What each arm isolates

- A vs B isolates the shared-experts sites. B passing where A failed means
  the shared-experts adapters were the damage: the shared expert fires on
  EVERY token regardless of routing, so a bad update there has the widest
  blast radius in the network.
- A vs C isolates the attention sites.
- B vs C head-to-head (if both run): which single path suffices for length
  reduction. B passing while C fails points at the attention output path
  (wo_b is the last linear before the attention residual; answer emission
  and termination are output-path behaviors) as the load-bearing site.

## Order to try

Default: B first, then C. Rationale: the cap's target behavior (emit the
answer, then stop) is an output-path behavior, and B keeps the always-on
FFN path pristine while halving the adapter budget. Stop at the first arm
that passes the same holdout gate. The winner re-enters the alpha sweep at
its own site set before any production claim.

## Cheap discriminator: per-site grad norms from the training log

Required change (no extra compute): tc_lora_train.py already computes every
adapter gradient each epoch in the reverse layer pass. Accumulate the L2
norm per site group (attn: wq_b/wkv/wo_b; experts: w1/w2/w3; optionally all
six sites separately) into report_train.json alongside the existing global
grad_norm. Read them off the FAILED A run's log before training anything
new:

| signal in A's log | prediction | first arm |
|---|---|---|
| expert-group norm >> attn-group (>~5x across epochs), loss plateaus | shared-experts sites absorb the update budget and are the likely accuracy damage | B |
| attn-group norm >> expert-group (wq_b spikes especially) | attention sites drive the damage | C |
| both small, loss flat | under-fitting: strength problem, not a site problem | none; raise alpha per alpha-sweep-plan.md instead |
| one site's norm explodes late (watch attn.wkv: it feeds the KV ring every layer and so edits attention content globally) | single-site over-reach | B variant minus that site (wq_b/wo_b only if wkv) |

Note: attn.wkv is the attention KV projection, not the separate compressor
wkv; editing it has the widest attention-side blast radius.

## Protocol invariants

- Step-0 identity check must still assert (B=0 adapter path reproduces the
  base forward exactly).
- Same paired holdout battery and sample seed as the failed A run; only the
  site set changes.
- One arm at a time, gate after each. Never change site set and alpha
  simultaneously: pick the site set first, then re-sweep alpha on the winner.
