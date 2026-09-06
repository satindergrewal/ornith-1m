#!/usr/bin/env python3
"""dsv4_phase4_plan.py - phase 4: inventory + declared venue + launch plan.

Port of the GLM-5.3 campaign's phase-4 driver (k35_phase4_plan.py in the
k35-dsv4-study port source, mirror cited per step) to the DSV4 Flash
campaign and the runpod 1-GPU encode venue. DSV4 deltas from the port
source (k35_phase4_plan.py:22-65):

  - the inventory is BUILT (or required) first: dsv4_build_inventory
    seals the packed-master walk under
    quant-pipeline.dsv4-release-inventory.v1; this driver runs it when
    inventory.json is absent (or --rebuild-inventory) and otherwise loads
    and re-verifies the sealed file through the plan builder's own
    surfaces check
  - the venue preflight is the declared sm120 document accepted by
    dsv4_uniform_k35 (worker count 1..4 from the document content): one
    RTX PRO 6000 Blackwell Server Edition today, workers=1, with the
    honest rationale "runpod 1-GPU venue; 4-worker when capacity"
  - NO baseline receipt is demanded (no uniform 4-bpw campaign exists;
    the launch plan carries an explicit declared-absent baseline block)
  - allocations cover layers 0..42 only (all routed; there is no MTP
    layer in the encode domain)
  - sealed sensitivity-DP allocations from the phase-5 probe driver are
    REUSED, not overwritten: re-running this driver after phase 5 is the
    re-plan step the encode worker names when a layer's disk allocation
    seal no longer matches the plan unit

Re-plan semantics: plan.json and state/state-0000.json are rewritten after
verification succeeds.  Superseded state files left in state/ are ignored
by the worker (its newest_state only considers states bound to the current
plan seal); move them to state/history/ for tidiness.

The declaration's runtime_receipt_sha256 links the venue to a measured
artifact: pass --runtime-receipt PATH (for example the CappedSource parity
verdict SOURCE_PARITY.json) to bind one; the default binds the inventory
seal, the only sealed artifact that exists at plan time on a fresh venue.

Usage (inside the encode container, cwd /wd):

  python3 dsv4_phase4_plan.py
  python3 dsv4_phase4_plan.py --work-root /workspace/dsv4-work

ASCII only. No em-dashes. No network. Writes only under --work-root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsv4_common as common
import dsv4_uniform_k35 as k35
from dsv4_common import die

WORK_ROOT_DEFAULT = os.environ.get("DSV4_WORK_ROOT", "/workspace/dsv4-work")
GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
COMPUTE_CAPABILITY = "12.0"
RATIONALE = "runpod 1-GPU venue; 4-worker when capacity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="phase 4 dsv4: inventory + declared preflight + launch plan"
    )
    parser.add_argument(
        "--work-root",
        default=WORK_ROOT_DEFAULT,
        type=Path,
        help="campaign work root (default env DSV4_WORK_ROOT or /workspace/dsv4-work)",
    )
    parser.add_argument(
        "--runtime-receipt",
        default=None,
        type=Path,
        help=(
            "venue runtime artifact whose sha256 backs the declaration "
            "(e.g. the CappedSource parity verdict SOURCE_PARITY.json); "
            "default: bind the inventory seal, the only sealed artifact at "
            "plan time"
        ),
    )
    parser.add_argument(
        "--rebuild-inventory",
        action="store_true",
        help="force dsv4_build_inventory to re-walk the packed master",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="capped-source construction device for the inventory build (default cpu)",
    )
    args = parser.parse_args()
    args.work_root = Path(args.work_root).resolve()
    return args


def load_or_build_inventory(args: argparse.Namespace) -> dict:
    """Inventory first: reuse the sealed file when present, else run the
    builder (DSV4 delta: the port source assumed a prebuilt BF16-master
    inventory, k35_phase4_plan.py:23; here the packed-master walk is part
    of the plan phase)."""
    path = args.work_root / "inventory.json"
    if args.rebuild_inventory or not path.is_file():
        from dsv4_build_inventory import build_inventory

        body = build_inventory(device=args.device)
        common.write_json(path, body)
        print("inventory built", body["inventory_sha256"][:16], flush=True)
    inventory = common.load_json(path)
    if not isinstance(inventory, dict):
        die(f"inventory is not an object: {path}")
    # Fail-closed consumer check (seal, geometry, scope census, shapes).
    main_rows, mtp_rows, _native_rows = k35._inventory_surfaces(inventory)
    print(
        f"inventory OK: routed={len(main_rows)} mtp={len(mtp_rows)} "
        f"checkpoint={inventory.get('checkpoint')}",
        flush=True,
    )
    return inventory


def build_preflight(inventory: dict, args: argparse.Namespace) -> dict:
    """Declared sm120 preflight for the runpod 1-GPU venue (mirror of
    k35_phase4_plan.py:26-45; one GPU row today, worker census honest)."""
    if args.runtime_receipt is not None:
        receipt_path = Path(args.runtime_receipt).resolve()
        if not receipt_path.is_file():
            die(f"runtime receipt is absent: {receipt_path}")
        runtime_receipt_sha256 = common.sha256_file(receipt_path)
    else:
        runtime_receipt_sha256 = str(inventory["inventory_sha256"])
    common.require_hash(
        runtime_receipt_sha256, "declared preflight runtime receipt link")
    return common.seal(
        {
            "schema": k35.SM120_DECLARED_PREFLIGHT_SCHEMA,
            "ready": True,
            "mode": "layer-streaming",
            "checkpoint_seal_mode": "full-shard-sha256",
            "checkpoint_inventory_sha256": inventory["inventory_sha256"],
            "workers": 1,
            "gpus": [
                {
                    "index": 0,
                    "name": GPU_NAME,
                    "compute_capability": COMPUTE_CAPABILITY,
                }
            ],
            "declaration": {
                "attested_by": k35.DECLARED_ATTESTED_BY,
                "rationale": RATIONALE,
                "runtime_receipt_sha256": runtime_receipt_sha256,
            },
        },
        "preflight_sha256",
    )


def load_or_seal_allocations(work_root: Path) -> dict[int, dict]:
    """Provisional allocations for layers 0..42 (deterministic_provisional
    basis), REUSING any sealed non-provisional phase-5 allocation on disk
    (the re-plan path; the port source always wrote provisional,
    k35_phase4_plan.py:47-53, because its probe phase followed planning)."""
    allocations: dict[int, dict] = {}
    provisional = 0
    sealed = 0
    directory = work_root / "allocations"
    directory.mkdir(parents=True, exist_ok=True)
    for layer in common.MAIN_LAYERS:
        path = directory / f"{common.probe_stem(layer)}.json"
        if path.is_file():
            receipt = common.load_json(path)
            if not isinstance(receipt, dict):
                die(f"existing allocation is not an object: {path}")
            # Fail-closed: a present-but-corrupt receipt aborts the re-plan
            # instead of being silently replaced.
            k35.verify_layer_allocation(receipt, layer=layer)
            if receipt.get("provisional") is False:
                allocations[layer] = receipt
                sealed += 1
                continue
            # A valid provisional receipt falls through and is re-sealed
            # (deterministic, so the rewrite is idempotent).
        receipt = k35.seal_layer_allocation(
            layer,
            k35.build_provisional_allocation(layer),
            provisional=True,
            basis="deterministic_provisional",
        )
        common.write_json(path, receipt)
        allocations[layer] = receipt
        provisional += 1
    return allocations


def main() -> int:
    args = parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)

    inventory = load_or_build_inventory(args)
    preflight = build_preflight(inventory, args)
    common.write_json(args.work_root / "preflight-declared.json", preflight)

    allocations = load_or_seal_allocations(args.work_root)
    provisional_count = sum(
        1 for receipt in allocations.values()
        if receipt.get("provisional") is True
    )
    print(
        f"allocations: {len(allocations)} layers "
        f"({provisional_count} provisional, "
        f"{len(allocations) - provisional_count} sealed sensitivity-DP)",
        flush=True,
    )

    plan = k35.build_launch_plan(
        inventory, preflight, layer_allocations=allocations)
    k35.verify_launch_plan(plan)
    common.write_json(args.work_root / "plan.json", plan)
    state = k35.initial_state(plan)
    state_dir = args.work_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(state_dir / "state-0000.json", state)
    superseded = sorted(
        path.name
        for path in state_dir.glob("state-*.json")
        if path.name != "state-0000.json"
    )
    print(
        json.dumps(
            {
                "plan": plan["launch_plan_sha256"][:16],
                "variant": plan["preflight_variant"],
                "workers": len(plan["scheduler"]["workers"]),
                "initial_queue_head": plan["scheduler"]["initial_queue"][:3],
                "provisional_allocations": provisional_count,
                "superseded_state_files_ignored": superseded,
            }
        )
    )
    print("PLAN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
