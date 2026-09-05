# Alpha sweep plan: DSV4 ThinkingCap pod run (cap strength)

Question: which LoRA alpha ships as the production cap. Three arms that
differ ONLY in alpha, each scored by the same holdout battery; the decision
comes from the gate numbers, not from training loss.

## Arms

| arm | R | alpha | scale (alpha/R) | source |
|-----|---|-------|-----------------|--------|
| a8  | 8 | 8     | 1.0             | train on pod |
| a16 | 8 | 16    | 2.0             | the 04-real baseline; reuse artifacts if they exist and the engine is unchanged |
| a32 | 8 | 32    | 4.0             | train on pod |

Alpha is the shipped variable: the merge folds scale into the weights
(W_eff = W + (alpha/R) * B@A), so this is an effective-update-strength sweep
(B starts at zero, scale multiplies B@A; at fixed LR, changing alpha changes
the adapter path's effective step size). R, LR, epochs frozen so alpha is
the only moving part. Keep per-arm output dirs (cap_a8/, cap_a16/, cap_a32/)
so artifacts never overwrite each other.

## Frozen across arms (one-difference rule)

- SFT rows: shortest-correct per problem from the merged regen set
- train seed 20260904; identical adapter init
- R=8, LR 3e-4, AdamW, 30 full-batch epochs, eos appended to every target,
  fp32 adapter path, layer-wise rematerialization
- holdout battery: seed 20260905, 12 problems (never regenerated), 3 samples
  per problem, temp 0.8, budget 160, official chat template, sample seed
  20260978 for EVERY arm including base -> paired samples; arms differ only
  by the folded cap
- gate = the tc_holdout_gate.py protocol, folded in memory; NO checkpoint
  merges during the sweep
- base arm: run once. Reuse the lane's existing base holdout report if the
  engine is unchanged; if the engine changed, re-run base once under the new
  engine and score every arm against that single run.

## Required plumbing (before any arm runs)

- tc_lora_train.py: alpha (and out dir) as CLI flags; they are module
  constants today.
- tc_holdout_gate.py capped arm: the fold scale must come from the arm's own
  alpha (alpha/R), not from the imported SCALE constant, or a8/a32 get
  scored at the wrong strength. Assert the artifact's recorded alpha matches
  the scale used to fold it.

## Metrics and gate (same battery, both arms paired)

Per arm vs base over the 36 holdout samples:
- accuracy delta = correct_first(capped) - correct_first(base)
- length delta = mean_tokens_correct(capped) - mean_tokens_correct(base)

Use the correct-only length mean. mean_tokens_all mixes in truncated
failures and reads wrong for this decision. Hit-cap counts are a secondary
signal only.

Gate per arm: accuracy delta >= -2 (within noise of 36 paired samples).
Length reduction is the objective, not a gate.

## Decision rule

1. Drop every arm failing the accuracy gate.
2. If no arm passes: there is NO production alpha. Do not pick the
   least-bad arm; go to ablation-fallback.md. (Check the training loss
   curves first: a diverging a32 loss predicts its gate failure and is a
   strength problem, not a site problem.)
3. Among passing arms, pick the LARGEST length reduction (most negative
   length delta).
4. Tie within 1 token: smaller alpha (the less intrusive cap).

The production run reuses the winning arm's artifacts; it is not retrained.

## Order and cost

1. plumbing flags (above)
2. base holdout (reuse if valid)
3. train a8 and a32 sequentially (a16 reused); record loss curve + grad norm
   per epoch
4. holdout capped arms a8, a16, a32, sequentially
5. compare and decide

Trainings are sequential (memory-bound rematerialization); each holdout arm
is one batched sampling run of at most 36 sequences x 160 tokens. Walls from
the node-a lane scale to the pod GPU; record actual walls in the results
table.

## Results table (fill on the pod)

| arm | train loss first->last | accuracy delta /36 | length delta (tok) | gate pass | notes |
|-----|------------------------|--------------------|--------------------|-----------|-------|
| a8  |                        |                    |                    |           |       |
| a16 |                        |                    |                    |           |       |
| a32 |                        |                    |                    |           |       |
