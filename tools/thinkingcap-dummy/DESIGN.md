# ThinkingCap DSV4 dummy cycle — nano-DSV4-Vision design record

Date: 2026-09-01 (NZ). Approved by Satinder 09-01 after model-selection discussion.

## Purpose

De-risk the ENTIRE ThinkingCap-on-DSV4 pipeline on the box for $0 before any paid pod
session. The dummy proves every script, config, and gate in the exact architecture and
format the real run will use. Big-model-only problems (real quality, memory scale,
multi-GPU, real drafter distributions) stay pod-side by design.

## Model selection (settled)

- Real target candidates: DeepSeek-V4-Flash-0731 (receipted training path via
  ms-swift/Megatron HF #40; drowzeys ablit base exists; matches our deployed Sparks
  serve) and the NEW deepseek-ai/DeepSeek-V4-Flash-Vision-Exp (4h old at decision time,
  stock/non-abliterated, vision modules, zero training receipts, ~199GB FP8+FP4,
  single config incl. vision tower, transformers 5.0.0).
- Satinder's preference: try cap-on-STOCK (like the canonical bottlecapai lineage),
  then apply community ablit when it lands and A/B whether cap-then-ablit survives.
- Principle recorded: the cap must be trained on-policy against the EXACT base you
  will deploy. cap-on-stock and cap-on-ablit both have receipts in the wild; MIXING
  orders (cap one base, ablate on top afterward) has ZERO receipts anywhere — that
  A/B is genuinely new territory, entered with eyes open.
- Dummy source config = Vision-Exp config (superset: text arch + vision tower frozen
  bystanders), shrunk field-by-field. Same dummy validates either future target.

## Nano config (field-by-field shrink of the real config)

Real -> nano: num_hidden_layers 43->4; num_nextn_predict_layers 3->3 (KEPT, the DSpark
drafter stack is the landmine surface); hidden_size 4096->256; num_attention_heads
64->8; head_dim 512->64; qk_rope_head_dim 64->16; num_key_value_heads 1->1;
q_lora_rank/o_lora_rank 1024->64; index_n_heads 64->4; index_head_dim 128->32;
index_topk 512->16; n_routed_experts 256->8; num_experts_per_tok 6->2; n_shared_experts
1->1; moe_intermediate_size 2048->64; num_hash_layers 3->1; o_groups 8->2; hc_mult 4
(kept); sliding_window 128->32; max_position_embeddings 1048576->4096 (no rope_scaling);
vision_n_layers 32->2; vision_dim 1024->64; vision_n_heads 16->4; vision_inter_dim
2816->128; vision_max_n_token 384->32; vocab_size 129280 (REAL tokenizer kept);
torch_dtype bfloat16; quantization_config OMITTED (the re-encode step must generate it
— that step IS part of what we are testing); expert_dtype omitted at source, produced
by the quant step.

## The cycle under test (all on box, all scripted)

1. Synthesize nano weights (BF16, random init) from the nano config with the real
   tokenizer files. Deliverable: loadable HF checkpoint.
2. Ablit-style direction step: compute refusal direction on nano, apply overlay
   (drowzeys-style FP8-overlay approach, on our BF16 nano it is a plain tensor edit).
   Records the script shape for the real run.
3. On-policy regen (hotdogs method): programmatically-verifiable problems (grade-school
   arithmetic), 3 samples from the CURRENT base, oracle filter, shortest-correct SFT +
   short-vs-verbose DPO. Model stays dumb; harness is the product.
4. Train: ms-swift or Megatron-Bridge LoRA on nano (language+expert layers; vision
   tower + nextn layers FROZEN per the drowzeys drafter-anchor lesson). Minutes/epoch.
5. Merge to BF16 checkpoint.
6. Re-encode: ModelOpt FP8/NVFP4-experts quant on the box 6000s (toolchain already
   proven here — the BF16-LMHead fix was this lane).
7. Gates: bake-serve merged nano on the day-0 vLLM image; greedy-equivalence,
   DSpark acceptance through the 3-layer nextn head, loop_eval, 3-arm comparison
   (stock vs knob vs cap).

## Sequencing

Config + scaffolding: no GPU, immediate. Train/quant/bench: behind the 3.5bpw encode
finish + morning bench (GPUs busy tonight). Dummy cycle target: complete through
Wednesday afternoon NZ.

## Honest non-goals

The dummy proves the PIPELINE, not the method's value on a real model. No quality
claims. Pod problems that remain: real quality preservation, real acceptance rates,
memory scale, multi-node.
