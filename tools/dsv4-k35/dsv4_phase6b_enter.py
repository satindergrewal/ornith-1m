#!/usr/bin/env python3
"""Phase 6b: enter the DSV4 k35 encoding phase (state-chain entry).

Port of k35_phase6b_enter.py (GLM-5.3 campaign snippet), which called
the SEALED state machine: glm53_uniform_k35.enter_k35_encoding
(glm53_uniform_k35.py:890-901; semantics: verify_state, refuse any
phase other than "planned", attach the readiness hash to evidence, then
write the sealed successor at sequence+1 per _successor
glm53_uniform_k35.py:880-888).

No sealed state machine exists for DSV4: this script carries the same
entry semantics over a NEW SURFACE state receipt
(quant-pipeline.dsv4-k35-state-receipt.v1).  The future phase-4/7 state
machine must adopt this schema, the phase names, and the evidence key
dsv4_readiness_receipt_sha256 (or this snippet must be re-pointed
before phase 7 runs).

Deliberate additions over the GLM snippet (both fail-closed): the
readiness receipt's launch_plan_sha256 must match the plan's seal, and
an existing successor state file is never overwritten.

Usage (cwd <work-root>, the same as the GLM snippet):

  python3 dsv4_phase6b_enter.py

ASCII only.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsv4_common as common
from dsv4_common import die

DSV4_STATE_SCHEMA = "quant-pipeline.dsv4-k35-state-receipt.v1"
PLANNED_PHASE = "planned"
ENCODING_PHASE = "k35_main_encoding"
READINESS_EVIDENCE_KEY = "dsv4_readiness_receipt_sha256"


def verify_state_seal(
    plan: Mapping[str, Any], state: Mapping[str, Any]
) -> str:
    """Structural DSV4 state verification (NEW SURFACE).

    Minimal mirror of glm53_uniform_k35.verify_state's entry checks
    (glm53_uniform_k35.py:773-878): seal, plan binding, sequence and
    predecessor discipline, phase vocabulary.  The scheduler partition
    and claim-receipt checks belong to the phase-4/7 state machine and
    are NOT reimplemented here; this snippet only ever enters from the
    initial "planned" state, which carries no claims or completions.
    """

    state_sha = common.verify_seal(
        state,
        schema=DSV4_STATE_SCHEMA,
        field="state_receipt_sha256",
        label="dsv4 state receipt",
    )
    if state.get("launch_plan_sha256") != plan["launch_plan_sha256"]:
        die("state receipt targets a different launch plan")
    sequence = state.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        die("state sequence is malformed")
    predecessor = state.get("previous_state_receipt_sha256")
    if sequence == 0 and predecessor is not None:
        die("initial state may not name a predecessor")
    if sequence > 0:
        common.require_hash(predecessor, "state predecessor receipt")
    if not isinstance(state.get("phase"), str):
        die("state phase is malformed")
    if not isinstance(state.get("evidence"), Mapping):
        die("state evidence is malformed")
    if (
        not isinstance(state.get("pending_layers"), list)
        or not isinstance(state.get("active_claims"), Mapping)
        or not isinstance(state.get("completed_layers"), Mapping)
    ):
        die("state scheduler domains are malformed")
    return state_sha


def enter_k35_encoding(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    readiness_receipt_sha256: str,
) -> dict[str, Any]:
    """Mirror of glm53_uniform_k35.enter_k35_encoding (:890-901)."""

    previous = verify_state_seal(plan, state)
    if state.get("phase") != PLANNED_PHASE:
        die(f"encoding may start only from {PLANNED_PHASE}")
    evidence = dict(state.get("evidence", {}))
    evidence[READINESS_EVIDENCE_KEY] = common.require_hash(
        readiness_receipt_sha256, "readiness receipt"
    )
    body = copy.deepcopy(dict(state))
    del body["state_receipt_sha256"]
    body["evidence"] = evidence
    body["phase"] = ENCODING_PHASE
    body["sequence"] = int(state["sequence"]) + 1
    body["previous_state_receipt_sha256"] = previous
    return common.seal(body, "state_receipt_sha256")


def main() -> None:
    plan = common.load_json("plan.json")
    if not isinstance(plan, Mapping):
        die("plan.json is not an object")
    # Schema-agnostic plan seal recompute (the DSV4 launch-plan schema
    # belongs to the future phase-4 driver; phase 6b binds only the seal
    # value, exactly as the GLM snippet did through the sealed machine).
    plan_sha = common.require_hash(
        plan.get("launch_plan_sha256"), "launch plan seal"
    )
    unsigned = dict(plan)
    del unsigned["launch_plan_sha256"]
    if common.sha256_bytes(common.canonical_json(unsigned)) != plan_sha:
        die("launch plan seal differs")

    state = common.load_json("state/state-0000.json")
    if not isinstance(state, Mapping):
        die("state/state-0000.json is not an object")

    readiness = common.load_json("gss/readiness-receipt.json")
    common.verify_seal(
        readiness,
        schema=common.DSV4_READINESS_SCHEMA,
        field="readiness_receipt_sha256",
        label="dsv4 readiness receipt",
    )
    if readiness.get("launch_plan_sha256") != plan_sha:
        die("readiness receipt binds a different launch plan")

    successor = enter_k35_encoding(
        plan, state,
        readiness_receipt_sha256=readiness["readiness_receipt_sha256"],
    )
    destination = Path(f"state/state-{successor['sequence']:04d}.json")
    if destination.exists():
        die(f"refusing to overwrite an existing state receipt: {destination}")
    destination.write_text(json.dumps(successor, indent=2) + "\n", encoding="utf-8")
    print("state", successor["sequence"], successor["phase"])


if __name__ == "__main__":
    main()
