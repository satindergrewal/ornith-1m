#!/usr/bin/env python3
"""gen_problems_podscale.py: pod-scale problem generator for the DSV4
ThinkingCap real-run campaign (Runpod prep; stdlib only, no torch/engine).

Scale-up of tc_batch_gen.make_problems for a future 500-2000 problem
campaign. Kinds stay 0/1/2:
    0: a + b            1: a * c            2: a + b * c
Kind-2 hard adds PEMDAS depth 2 (a + b * c * d: a second factor in the
product term), so the answer is still a single integer and
tc_batch_gen.parse_answer last-integer grading is unchanged. No division,
no parentheses, no negatives.

Seed discipline (HARD RULES):
  - Holdout seed 20260905 is REFUSED as --seed. The holdout battery
    (12 problems, ../data/holdout_problems.json) must never be regenerated
    or trained on.
  - --exclude-holdout reads the holdout file if present, parses each q back
    to its expression, and skips any generated collision. The holdout seed
    is never replayed through this generator (its RNG stream belongs to the
    legacy make_problems draw order; replaying it here would be both wrong
    and a holdout touch).

Differences from tc_batch_gen.make_problems (intentional, documented):
  - Only the operands a kind uses are drawn (legacy drew a,b,c and ignored
    one per kind), so a given seed produces a different stream than legacy.
  - Operand ranges come from the difficulty tier, not fixed 2-49 / 2-9.
  - Dedup key is (kind, expression): on legacy-shaped problems this is
    equivalent to (a, b, c, kind), and it stays correct for kind-2 depth 2
    where a bare (a, b, c, kind) key would both false-collide and miss.
    Exact-match only: commuted near-dups ("3 + 5" vs "5 + 3") are NOT
    collapsed.
  - Deterministic: same args -> identical output (single rng, fixed
    kind x difficulty iteration order).

Rows (JSONL, one per problem): pid, kind, difficulty, seed, operand fields
(a,b for kind 0; a,c for kind 1; a,b,c plus d when depth 2 for kind 2),
depth (kind 2 only), q, ans, gen.

Usage:
  python3 gen_problems_podscale.py --n 1000 --seed 20260910 \
      [--mix 1:1:1] [--kinds 0,1,2] [--out pod_problems.jsonl] [--exclude-holdout]
  python3 gen_problems_podscale.py --selftest    # 20 problems, writes nothing
"""

import argparse
import json
import os
import random

HOLDOUT_SEED = 20260905   # refused as a campaign seed, never regenerated here
HOLDOUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "holdout_problems.json")
GEN_VERSION = "podscale/1"
DIFFS = ("easy", "medium", "hard")
INTENDED_RANGE = (500, 2000)

# Tier specs: operand digit counts in draw order (legacy variable names:
# kind 0 draws a,b; kind 1 draws a,c; kind 2 draws a,b,c then d at depth 2).
# kind-2 depth = number of factors in the product term (1 = legacy a + b*c).
TIERS = {
    0: {"easy":   {"digits": (1, 1)},
        "medium": {"digits": (2, 2)},
        "hard":   {"digits": (3, 3)}},
    1: {"easy":   {"digits": (1, 1)},
        "medium": {"digits": (2, 1)},
        "hard":   {"digits": (2, 2)}},
    2: {"easy":   {"digits": (1, 1, 1), "depth": 1},
        "medium": {"digits": (2, 1, 1), "depth": 1},
        "hard":   {"digits": (2, 2, 1, 1), "depth": 2}},
}


def check_tiers():
    """Table sanity: digit-count arity matches each kind's operand count."""
    for kind, tiers in TIERS.items():
        for diff, spec in tiers.items():
            depth = spec.get("depth", 1)
            want = 2 if kind in (0, 1) else 2 + depth
            if len(spec["digits"]) != want:
                raise SystemExit(f"bad tier kind {kind}/{diff}: "
                                 f"{len(spec['digits'])} digit specs, want {want}")


def digit_span(d):
    """Inclusive operand range for a d-digit tier. 1-digit keeps the legacy
    minimum of 2 (no a*1 / a+0 trivialities)."""
    lo = 2 if d == 1 else 10 ** (d - 1)
    return lo, 10 ** d - 1


def render(kind, ops):
    """Operands in draw order -> (expression, answer). Depth-1 kind-2 output
    is byte-identical to the legacy make_problems surface."""
    if kind == 0:
        a, b = ops
        return f"{a} + {b}", a + b
    if kind == 1:
        a, c = ops
        return f"{a} * {c}", a * c
    prod = 1
    for x in ops[1:]:
        prod *= x
    return f"{ops[0]} + " + " * ".join(str(x) for x in ops[1:]), ops[0] + prod


def expr_of_q(q):
    """'What is <expr>?' -> '<expr>' (both generators emit this exact shape)."""
    pre = "What is "
    if q.startswith(pre) and q.endswith("?"):
        return q[len(pre):-1]
    return None


def load_holdout_keys():
    """Holdout exclusion keys = expression strings parsed from the holdout
    file. File is the authority (it is what the gate battery reads). Missing
    file -> None (caller warns; exclusion becomes a no-op; the seed is NOT
    replayed)."""
    if not os.path.exists(HOLDOUT_PATH):
        return None
    with open(HOLDOUT_PATH) as f:
        d = json.load(f)
    exprs = set()
    for p in d.get("problems", []):
        e = expr_of_q(p["q"])
        if e is not None:
            exprs.add(e)
    return exprs


def is_holdout_seed(seed):
    return seed == HOLDOUT_SEED


def allocate(total, weights):
    """Largest-remainder split of total over nonnegative weights
    (deterministic; ties resolved by index order)."""
    if total < 0 or sum(weights) <= 0:
        raise SystemExit(f"bad allocation: total={total} weights={weights}")
    raw = [total * w / sum(weights) for w in weights]
    out = [int(x) for x in raw]
    order = sorted(range(len(weights)), key=lambda j: raw[j] - out[j],
                   reverse=True)
    for j in order[: total - sum(out)]:
        out[j] += 1
    return out


def draw_cell(rng, kind, diff, n, seen, holdout_exprs, seed, stats):
    """Rejection-sample n problems for one (kind, difficulty) cell."""
    spec = TIERS[kind][diff]
    depth = spec.get("depth", 1)
    spans = [digit_span(d) for d in spec["digits"]]
    space = 1
    for lo, hi in spans:
        space *= hi - lo + 1
    if n > space:
        raise SystemExit(
            f"infeasible: kind {kind}/{diff} wants {n} problems but the "
            f"exact-match space is {space} ({spec['digits']}-digit operands); "
            f"lower --n or shift --mix away from this cell")
    rows, attempts = [], 0
    while len(rows) < n:
        attempts += 1
        if attempts > 100 * n + 1000:
            raise SystemExit(f"kind {kind}/{diff}: rejection sampling stuck "
                             f"({len(rows)}/{n} after {attempts} draws)")
        ops = [rng.randint(lo, hi) for lo, hi in spans]
        expr, ans = render(kind, ops)
        if holdout_exprs is not None and expr in holdout_exprs:
            stats["holdout_skip"] += 1
            continue
        if (kind, expr) in seen:
            stats["dup_skip"] += 1
            continue
        seen.add((kind, expr))
        row = {"kind": kind, "difficulty": diff}
        if kind == 0:
            row["a"], row["b"] = ops
        elif kind == 1:
            row["a"], row["c"] = ops
        else:
            row["a"], row["b"], row["c"] = ops[0], ops[1], ops[2]
            row["depth"] = depth
            if depth > 1:
                row["d"] = ops[3]
        row["q"], row["ans"] = f"What is {expr}?", ans
        rows.append(row)
    stats["attempts"] += attempts
    return rows


def main():
    check_tiers()
    ap = argparse.ArgumentParser(
        description="Pod-scale ThinkingCap problem generator (kinds 0/1/2, "
                    "difficulty tiers, dedup, holdout exclusion).")
    ap.add_argument("--n", type=int, default=1000,
                    help=f"problem count (intended {INTENDED_RANGE[0]}-"
                         f"{INTENDED_RANGE[1]}; default %(default)s)")
    ap.add_argument("--seed", type=int, default=None,
                    help="campaign problem seed (required for real runs; "
                         f"{HOLDOUT_SEED} is refused)")
    ap.add_argument("--kinds", default="0,1,2",
                    help="subset of 0,1,2 (default %(default)s)")
    ap.add_argument("--mix", default="1:1:1",
                    help="easy:medium:hard ratio (default %(default)s)")
    ap.add_argument("--out", default="pod_problems.jsonl",
                    help="output JSONL (default %(default)s)")
    ap.add_argument("--exclude-holdout", action="store_true",
                    help="skip collisions with ../data/holdout_problems.json")
    ap.add_argument("--selftest", action="store_true",
                    help="generate 20 problems, print stats, write nothing")
    a = ap.parse_args()

    n = 20 if a.selftest else a.n
    if n < 1:
        ap.error(f"--n must be >= 1 (got {a.n})")
    if a.seed is None:
        if a.selftest:
            a.seed = 1
        else:
            ap.error("--seed is required (holdout seed "
                     f"{HOLDOUT_SEED} is refused)")
    if is_holdout_seed(a.seed):
        ap.error(f"--seed {a.seed} is the HOLDOUT seed; it must never be "
                 f"used for a campaign set (battery: {HOLDOUT_PATH})")
    if not a.selftest and not (INTENDED_RANGE[0] <= n <= INTENDED_RANGE[1]):
        print(f"[warn] --n {n} outside the intended campaign range "
              f"{INTENDED_RANGE[0]}-{INTENDED_RANGE[1]}; proceeding")

    try:
        kinds = sorted({int(k) for k in a.kinds.split(",")})
    except ValueError:
        kinds = []
    if not kinds or any(k not in (0, 1, 2) for k in kinds):
        ap.error(f"--kinds must be a subset of 0,1,2 (got {a.kinds!r})")
    try:
        mix = [int(x) for x in a.mix.split(":")]
    except ValueError:
        mix = []
    if len(mix) != 3 or any(w < 0 for w in mix) or sum(mix) == 0:
        ap.error(f"--mix must be three nonneg ints easy:medium:hard with a "
                 f"positive sum (got {a.mix!r})")

    holdout_exprs = None
    if a.exclude_holdout or a.selftest:
        holdout_exprs = load_holdout_keys()
        if holdout_exprs is None:
            print(f"[warn] holdout file not found ({HOLDOUT_PATH}); "
                  f"exclusion is a no-op (holdout seed NOT replayed)")
        else:
            print(f"[holdout] {len(holdout_exprs)} expressions loaded "
                  f"for collision exclusion")
    elif os.path.exists(HOLDOUT_PATH):
        print("[warn] holdout file present but --exclude-holdout not set: "
              "generated problems are NOT checked against the holdout battery")

    rng = random.Random(a.seed)
    seen = set()
    stats = {"dup_skip": 0, "holdout_skip": 0, "attempts": 0}
    rows = []
    for k, nk in zip(kinds, allocate(n, [1] * len(kinds))):
        # Tier spill: small cells (e.g. 1-digit x 1-digit = 64 exact-match
        # expressions) saturate below a large --n; the surplus moves to the
        # NEXT tier of the same kind (easy->medium->hard) instead of failing.
        # take is capped 8 under the raw space so holdout collisions cannot
        # wedge the rejection sampler when drawing near-capacity.
        spill = 0
        spilled = []
        for diff, cnt in zip(DIFFS, allocate(nk, mix)):
            want = cnt + spill
            spill = 0
            if not want:
                continue
            spec = TIERS[k][diff]
            space = 1
            for d in spec["digits"]:
                lo, hi = digit_span(d)
                space *= hi - lo + 1
            take = min(want, max(space - 8, 0))
            if take < want:
                spill = want - take
                spilled.append((diff, spill))
            if take:
                rows += draw_cell(rng, k, diff, take, seen, holdout_exprs,
                                  a.seed, stats)
        if spill:
            raise SystemExit(
                f"infeasible: kind {k} still holds {spill} unplaced problems "
                f"after tier spill (all tiers saturated); lower --n")
    assert len(rows) == n, f"row count {len(rows)} != requested {n}"
    assert len({r["q"] for r in rows}) == n, "duplicate q after dedup"
    final = []
    for i, r in enumerate(rows):
        kind, diff = r.pop("kind"), r.pop("difficulty")
        r["gen"] = GEN_VERSION
        final.append({"pid": i, "kind": kind, "difficulty": diff,
                      "seed": a.seed, **r})

    print(f"[gen] n={n} seed={a.seed} kinds={kinds} mix={a.mix} "
          f"-> {len(final)} problems")
    for k in kinds:
        cnts = {d: sum(1 for r in final
                       if r["kind"] == k and r["difficulty"] == d)
                for d in DIFFS}
        print(f"       kind {k}: " +
              " ".join(f"{d}={cnts[d]}" for d in DIFFS))
    print(f"[gen] dedup: {stats['dup_skip']} duplicate draws skipped, "
          f"{stats['holdout_skip']} holdout collisions skipped, "
          f"{stats['attempts']} total draws")
    for r in final[:6]:
        extra = f" depth={r['depth']}" if "depth" in r else ""
        print(f"       pid{r['pid']} k{r['kind']} {r['difficulty']}{extra}: "
              f"{r['q']} = {r['ans']}")

    if a.selftest:
        print(f"[selftest] seed guard: refusing --seed {HOLDOUT_SEED}: "
              f"{is_holdout_seed(HOLDOUT_SEED)}")
        print("[selftest] no files written")
        return 0
    with open(a.out, "w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")
    print(f"[gen] wrote {a.out} ({os.path.getsize(a.out)} bytes, "
          f"{len(final)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
