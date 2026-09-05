#!/usr/bin/env python3
"""build_merged_report.py — merge the 48-budget kind-0/1 run with the
160-budget kind-2 run into one regen.jsonl with provenance, and regenerate
regen_report.json (per-kind yield + mean completion length + finished-before-
cap vs hit-cap), per coordinator order 2026-09-04."""
import json
import re


def first_answer(text):
    m = re.findall(r"(?:=|\bis\b|\*\*)\s*\**\s*(-?\d+)", text.split("\n")[0])
    return int(m[0]) if m else None


def enrich(row, budget):
    row["kind"] = row["pid"] % 3
    row["budget"] = budget
    row["hit_cap"] = row["n_new"] >= budget
    row.setdefault("correct_first_answer", first_answer(row["completion"]) == row["answer"])
    return row


def main():
    old = [json.loads(l) for l in open("/wd/data/regen_run48.jsonl")]
    new = [json.loads(l) for l in open("/wd/data/regen_kind2.jsonl")]
    rows = [enrich(r, 48) for r in old if r["pid"] % 3 != 2] + \
           [enrich(r, 160) for r in new]
    rows.sort(key=lambda r: (r["pid"], r["sample"]))
    with open("/wd/data/regen.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    per_kind = {}
    for k in (0, 1, 2):
        rs = [r for r in rows if r["kind"] == k]
        ok = [r for r in rs if r["correct"] or r["correct_first_answer"]]
        per_kind[str(k)] = {
            "label": ["a+b", "a*c", "a+b*c"][k],
            "budget": rs[0]["budget"] if rs else None,
            "samples": len(rs),
            "correct_strict": sum(r["correct"] for r in rs),
            "correct_first": sum(r["correct_first_answer"] for r in rs),
            "problems_with_correct": len({r["pid"] for r in ok}),
            "hit_cap": sum(r["hit_cap"] for r in rs),
            "finished_before_cap": sum(not r["hit_cap"] for r in rs),
            "mean_completion_tokens_all": round(
                sum(r["n_new"] for r in rs) / max(len(rs), 1), 1),
            "mean_completion_tokens_correct": round(
                sum(r["n_new"] for r in ok) / max(len(ok), 1), 1),
        }
    pids_ok = {r["pid"] for r in rows
               if r["correct"] or r["correct_first_answer"]}
    report = {
        "merged": True, "n_rows": len(rows),
        "sources": {"kind0/1": "regen_run48.jsonl (budget 48)",
                    "kind2": "regen_kind2.jsonl (budget 160)"},
        "correct_strict": sum(r["correct"] for r in rows),
        "correct_first": sum(r["correct_first_answer"] for r in rows),
        "problems_with_correct": len(pids_ok),
        "n_problems": len({r["pid"] for r in rows}),
        "per_kind": per_kind,
    }
    json.dump(report, open("/wd/data/regen_report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))
    # SFT view: shortest-correct per problem
    best = {}
    for r in rows:
        if r["correct"] or r["correct_first_answer"]:
            if r["pid"] not in best or r["n_new"] < best[r["pid"]]["n_new"]:
                best[r["pid"]] = r
    sft = [{"pid": r["pid"], "kind": r["kind"], "prompt": r["prompt"],
            "completion": r["completion"], "n_new": r["n_new"],
            "budget": r["budget"]} for r in
           (best[p] for p in sorted(best))]
    with open("/wd/data/sft.jsonl", "w") as f:
        for r in sft:
            f.write(json.dumps(r) + "\n")
    print(f"SFT rows (shortest-correct per problem): {len(sft)}; "
          f"kind mix: {sorted(r['kind'] for r in sft).count(0)}/"
          f"{sorted(r['kind'] for r in sft).count(1)}/{sorted(r['kind'] for r in sft).count(2)}")


if __name__ == "__main__":
    main()
