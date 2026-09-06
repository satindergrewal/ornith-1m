#!/usr/bin/env python3
"""dsv4_phase8_selftest.py - off-pod selftest for dsv4_surface.py and
dsv4_phase8_pack.py.

What runs here and what deliberately does not:

- A synthetic master-metadata tree (config.json + tensor_meta.json with the
  FULL discovered name census, plus a 258-site cap adapter) lets
  dsv4_common construct the real Geometry exactly as it does on the pod;
  every census assert stays real.
- The synthetic LAYER STORE is shrunk: the campaign census constants on
  dsv4_common are patched down (2 experts, 6 tensors per layer, 3 K4, 21
  bit units) so a full receipt chain (2 expert receipts x 3 sealed
  choices, real 2048x4096 FP16 reconstructions) fits in ~150MB. The
  production modules read these constants dynamically (module attribute
  access), which is what makes the patch meaningful rather than a lie:
  every census check still runs, against the patched census.
- NOT exercised off-pod (infeasible without the pod's packed master):
  load_dsv4_surface's plan/state/readiness binding (would need a sealed
  inventory over the real ~130GB master), the native/mtp dequant paths
  (CappedSource needs the pod full_loader), and the full
  materialize_checkpoint/verify_packed_checkpoint closure.
- Exercised: layer chain verification + mixed-rate census + fail-closed
  negatives; per-shard materialization with a safetensors round-trip,
  resumable .done markers, and index total_size accounting.

Run: /path/to/python3.11 dsv4_phase8_selftest.py [--keep]
(the python needs torch, safetensors, numpy; the quant_pipeline package
from the glm53-quant-repo src tree is expected at /tmp/glm53repo or
override with --glm-src PATH).

ASCII only. No em-dashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent


def build_synthetic_master(root: Path) -> tuple[Path, Path, Path]:
    """Master metadata census good enough for dsv4_geometry.Geometry.

    Names and dtypes follow the discovered grammar; no shard bytes are
    written (Geometry never reads shard files, only dsv4_build_inventory
    does, and the inventory path is not exercised here).
    """
    model_dir = root / "model"
    model_dir.mkdir(parents=True)
    config = {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_hidden_layers": 43,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 2048,
        "hidden_size": 4096,
        "num_nextn_predict_layers": 3,
        "routed_scaling_factor": 1.5,
        "norm_topk_prob": True,
        "vocab_size": 1024,
        "quantization_config": {"quant_method": "fictional-fp8-block"},
    }
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=1), encoding="utf-8")
    meta: dict[str, list] = {}
    shard = "toy-shard-000.safetensors"
    for layer in range(43):
        for expert in range(256):
            for projection in ("w1", "w2", "w3"):
                meta[f"layers.{layer}.ffn.experts.{expert}."
                     f"{projection}.weight"] = ["I8", [2048, 2048], shard]
    # mtp census the geometry asserts: mtp.{0,1,2} x 256 experts x 3
    for module in range(3):
        for expert in range(256):
            for projection in ("w1", "w2", "w3"):
                meta[f"mtp.{module}.ffn.experts.{expert}."
                     f"{projection}.weight"] = ["I8", [2048, 2048], shard]
    for layer in range(3):
        meta[f"layers.{layer}.ffn.gate.tid2eid.weight"] = [
            "I64", [1024, 6], shard]
    for layer in range(43):
        for site in (
            "attn.wq_b", "attn.wkv", "attn.wo_b",
            "ffn.shared_experts.w1", "ffn.shared_experts.w2",
            "ffn.shared_experts.w3",
        ):
            meta[f"layers.{layer}.{site}.weight"] = [
                "F8_E4M3", [64, 64], shard]
            meta[f"layers.{layer}.{site}.scale"] = [
                "F8_E8M0", [64], shard]
    meta_path = root / "tensor_meta.json"
    meta_path.write_text(
        json.dumps({"meta": meta}, indent=1), encoding="utf-8")

    import torch
    from safetensors.torch import save_file

    lora: dict[str, torch.Tensor] = {}
    for layer in range(43):
        for site in (
            "attn.wq_b", "attn.wkv", "attn.wo_b",
            "ffn.shared_experts.w1", "ffn.shared_experts.w2",
            "ffn.shared_experts.w3",
        ):
            for key in ("A", "B"):
                lora[f"lora.layers.{layer}.{site}.weight.{key}"] = (
                    torch.zeros(1, 1, dtype=torch.float16))
    lora_path = root / "lora.safetensors"
    save_file(lora, str(lora_path))
    return model_dir, meta_path, lora_path


def hex64(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--glm-src", default="/tmp/glm53repo/glm53-quant-repo/src", type=Path)
    args = parser.parse_args()

    root = Path(tempfile.mkdtemp(prefix="dsv4-selftest-"))
    if not args.keep:
        import atexit

        atexit.register(shutil.rmtree, root, ignore_errors=True)

    model_dir, meta_path, lora_path = build_synthetic_master(root)
    import os

    os.environ["DSV4_MODEL"] = str(model_dir)
    os.environ["DSV4_META"] = str(meta_path)
    os.environ["DSV4_LORA"] = str(lora_path)
    sys.path.insert(0, str(TOOLS))
    sys.path.insert(0, str(args.glm_src))

    import torch

    import dsv4_common as common
    import dsv4_uniform_k35 as k35_real  # validates the REAL census first

    # Shrink the campaign census for the synthetic layer store (documented
    # in the module docstring; production code reads these dynamically).
    # dsv4_uniform_k35 must already be imported: its import-time
    # validate_rate_arithmetic asserts the real 768/2688 census and would
    # reject the patched values.
    common.NUM_EXPERTS = 2
    common.TENSORS_PER_LAYER = 6
    common.K4_TENSORS_PER_LAYER = 3
    common.K3_TENSORS_PER_LAYER = 3
    common.TARGET_BIT_UNITS_PER_LAYER = 21

    import dsv4_phase8_pack
    import dsv4_surface
    import dsv4_worker

    work = root / "work"
    layers_root = work / "layers"
    layer_root = layers_root / "L00"
    store = common.Dsv4PackedPayloadStore(layer_root / "payload-store")
    k35 = k35_real  # the patched-census view of the same module

    claim_sha = hex64("selftest-claim")
    reader_abi = hex64("selftest-reader-abi")
    codec_identity = hex64("selftest-codec-identity")

    # bits per (expert, projection): exactly 3 K4 tensors, 21 bit units.
    bits_map = {
        (0, "w1"): 4, (0, "w2"): 4, (0, "w3"): 3,
        (1, "w1"): 3, (1, "w2"): 4, (1, "w3"): 3,
    }
    allocation = {
        common.tensor_full_name(0, expert, projection): bits
        for (expert, projection), bits in bits_map.items()
    }
    allocation_receipt = k35.seal_layer_allocation(
        0, allocation, provisional=False, basis="selftest_synthetic")
    alloc_sha = allocation_receipt["allocation_sha256"]

    logical = common.G.logical_shapes
    originals: dict[tuple, dict[str, torch.Tensor]] = {}
    expert_choices: dict[int, dict[str, dict]] = {}
    torch.manual_seed(7)
    for expert in range(common.NUM_EXPERTS):
        predecessor = claim_sha
        choices: dict[str, dict] = {}
        # encode order: w1 -> w3 -> w2 (dsv4_worker.py:1014-1065, 1112-1186)
        for projection in ("w1", "w3", "w2"):
            bits = bits_map[(expert, projection)]
            n, k = (
                logical[projection] if projection in ("w1", "w3")
                else (logical["w2"][0], logical["w2"][1]))
            tensors = {
                "reconstruction": torch.randn(
                    n, k, dtype=torch.float16),
                "trellis": torch.zeros(
                    n * k * bits // 8 // 2, dtype=torch.int16),
                "suh": torch.randn(k, dtype=torch.float16),
                "svh": torch.randn(n, dtype=torch.float16),
                "mcg": torch.tensor(
                    [dsv4_surface.MCG_MARKER_SIGNED_INT32],
                    dtype=torch.int32),
            }
            choice = store.put_choice(
                layer=0,
                expert=expert,
                projection=projection,
                bits=bits,
                choice_id=dsv4_surface.expected_choice_id(
                    0, expert, projection, bits),
                trellis=tensors["trellis"],
                suh=tensors["suh"],
                svh=tensors["svh"],
                mcg=tensors["mcg"],
                reconstruction=tensors["reconstruction"],
                vector_topology={"suh": "layer_shared", "svh": "expert_private"},
                reader_abi_sha256=reader_abi,
                provenance={
                    "claim_receipt_sha256": claim_sha,
                    "allocation_sha256": alloc_sha,
                    "bits": bits,
                    "vector_rate": bits,
                },
                predecessor_state_hash=predecessor,
            )
            predecessor = choice["choice_sha256"]
            choices[projection] = choice
            originals[(expert, projection)] = tensors
        expert_choices[expert] = choices

    expert_receipts = []
    # dsv4_worker relies on the layer tree already existing for its receipt
    # writes (its payload store only creates its own dirs); create the
    # experts dir here the way the encode runbook's layer bootstrap does.
    (layer_root / "experts" / common.layer_dir_name(0)).mkdir(
        parents=True, exist_ok=True)
    for expert in range(common.NUM_EXPERTS):
        receipt = dsv4_worker.build_expert_receipt(
            layer=0,
            expert=expert,
            bits_by_projection={
                p: bits_map[(expert, p)] for p in common.PROJECTIONS},
            choices=expert_choices[expert],
            claim_receipt_sha256=claim_sha,
            allocation_sha256=alloc_sha,
            capture_binding={
                "schema": common.DSV4_CAPTURE_SCHEMA,
                "capture_sha256": hex64("capture"),
                "layer": 0,
                "rows_per_layer": 8,
                "roles": ["fit", "conditional-fit"],
                "calibration_root": str(root),
            },
            hessian_artifact={
                "path": str(layer_root / "hessians" / "layer-00" / "x.safetensors"),
                "bytes": 1,
                "sha256": hex64("hessian"),
            },
            down_conditioning={
                "gate_rate": 3,
                "up_rate": 3,
                "down_rate": bits_map[(expert, "w2")],
                "deployed_gate_rate": bits_map[(expert, "w1")],
                "deployed_up_rate": bits_map[(expert, "w3")],
                "semantics": "r7_pair_at_reference_rates_v1",
                "evidence": {
                    "conditioning_gate_bits": 3,
                    "conditioning_up_bits": 3,
                    "rows": 4,
                },
            },
            codec_identity_sha256=codec_identity,
        )
        path = (
            layer_root / "experts" / common.layer_dir_name(0)
            / f"expert-{expert:03d}.json")
        common.write_json(path, receipt)
        expert_receipts.append(receipt)

    layer_receipt = dsv4_worker.build_layer_receipt(
        layer=0,
        worker_id="sm120-0",
        claim_receipt_sha256=claim_sha,
        allocation_sha256=alloc_sha,
        expert_receipts=expert_receipts,
    )
    common.write_json(layer_root / "layer-receipt.json", layer_receipt)

    # ------------------------------------------------------------------
    # Surface: chain verification, census, fail-closed negatives.
    # ------------------------------------------------------------------
    chain = dsv4_surface.verify_layer_chain(
        layers_root, 0, expected_allocation_sha256=alloc_sha)
    assert chain["receipt_sha256"] == layer_receipt["receipt_sha256"]
    assert len(chain["choices"]) == common.TENSORS_PER_LAYER

    surface = dsv4_surface.Dsv4Surface(
        root=layers_root,
        choices=chain["choices"],
        layer_receipt_sha256=(layer_receipt["receipt_sha256"],),
        layer_receipts=(layer_receipt,),
        store=dsv4_surface.Dsv4MultiLayerStore(layers_root),
        packed_reader_abi_sha256=reader_abi,
    )
    census = surface.rate_census()
    assert census == {"k3_choice_count": 3, "k4_choice_count": 3}, census
    assert surface.choice(0, 1, "w2")["bits"] == 4

    def expect_die(label, thunk):
        try:
            thunk()
        except SystemExit:
            print(f"  fail-closed OK: {label}")
            return
        raise AssertionError(f"fail-closed check did not die: {label}")

    expect_die(
        "foreign allocation expectation",
        lambda: dsv4_surface.verify_layer_chain(
            layers_root, 0, expected_allocation_sha256=hex64("foreign")))
    tampered = json.loads(json.dumps(layer_receipt))
    tampered["bits"] = "uniform_k4"
    expect_die(
        "tampered layer receipt marker",
        lambda: dsv4_surface.verify_layer_chain(
            layers_root, 0, receipt=tampered))
    expect_die(
        "absent layer store",
        lambda: dsv4_surface.Dsv4MultiLayerStore(
            layers_root).store_for(3))
    # Tampered choice file on disk (the store copy differs from the expert
    # receipt's embedded copy): the chain must die.
    choice_digest = expert_choices[0]["w1"]["choice_sha256"]
    choice_path = (
        layer_root / "payload-store" / "choices" / f"{choice_digest}.json")
    pristine_choice = choice_path.read_text(encoding="utf-8")
    tampered_choice = json.loads(pristine_choice)
    tampered_choice["bits"] = 3
    choice_path.write_text(
        json.dumps(tampered_choice, indent=1), encoding="utf-8")
    expect_die(
        "tampered store choice file",
        lambda: dsv4_surface.verify_layer_chain(layers_root, 0))
    choice_path.write_text(pristine_choice, encoding="utf-8")
    print("surface chain + census OK")

    # ------------------------------------------------------------------
    # Pack: 2-tensor materialization, round-trip, resume, index.
    # ------------------------------------------------------------------
    rows = [
        {
            "tensor_name": "layers.0.ffn.experts.0.w1.weight",
            "scope": "routed_expert",
            "dtype": "I8",
            "shape": [2048, 2048],
            "source_bytes": 1,
            "source_payload_sha256": hex64("src0"),
            "shard": "toy-00001-of-00001.safetensors",
        },
        {
            "tensor_name": "layers.0.ffn.experts.1.w2.weight",
            "scope": "routed_expert",
            "dtype": "I8",
            "shape": [4096, 1024],
            "source_bytes": 1,
            "source_payload_sha256": hex64("src1"),
            "shard": "toy-00001-of-00001.safetensors",
        },
    ]
    out_root = root / "out"
    plan_sha = hex64("selftest-pack-plan")
    shard_receipt = dsv4_phase8_pack.materialize_shard(
        pack_plan_sha256=plan_sha,
        source=None,
        output_root=out_root,
        source_shard="toy-00001-of-00001.safetensors",
        shard_rows=rows,
        surface=surface,
    )
    assert shard_receipt["complete"] is True
    assert shard_receipt["routed_choice_count"] == 2
    assert shard_receipt["output_tensor_count"] == 8
    shard_path = out_root / "toy-00001-of-00001.safetensors"
    assert shard_path.is_file()
    done = dsv4_phase8_pack._shard_done_path(
        out_root, "toy-00001-of-00001.safetensors")
    assert done.is_file()
    assert done.read_text(encoding="utf-8").strip() == (
        shard_receipt["receipt_sha256"])

    from safetensors import safe_open

    expected_bytes = 0
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {
            record["name"] for record in shard_receipt["tensors"]}
        for (expert, projection), tensors in originals.items():
            if (expert, projection) not in {(0, "w1"), (1, "w2")}:
                continue
            for suffix in dsv4_phase8_pack.PACKED_SUFFIXES:
                name = dsv4_phase8_pack.packed_tensor_name(
                    common.tensor_full_name(0, expert, projection), suffix)
                value = handle.get_tensor(name)
                assert torch.equal(value, tensors[suffix].contiguous()), name
        for record in shard_receipt["tensors"]:
            expected_bytes += int(record["bytes"])
    print("materialize + safetensors round-trip OK")

    # Resume: the .done marker short-circuits without touching the file.
    stamp = shard_path.stat().st_mtime_ns
    resumed = dsv4_phase8_pack.materialize_shard(
        pack_plan_sha256=plan_sha,
        source=None,
        output_root=out_root,
        source_shard="toy-00001-of-00001.safetensors",
        shard_rows=rows,
        surface=surface,
    )
    assert resumed == shard_receipt
    assert shard_path.stat().st_mtime_ns == stamp
    # A stale .done marker (crash between receipt and marker) re-derives.
    done.write_text(hex64("stale") + "\n", encoding="utf-8")
    expect_die(
        "stale .done marker",
        lambda: dsv4_phase8_pack.materialize_shard(
            pack_plan_sha256=plan_sha,
            source=None,
            output_root=out_root,
            source_shard="toy-00001-of-00001.safetensors",
            shard_rows=rows,
            surface=surface,
        ))
    done.write_text(
        shard_receipt["receipt_sha256"] + "\n", encoding="utf-8")
    # A shard file with neither receipt nor .done (crash after the atomic
    # shard write) is rewritten, never trusted in place.
    done.unlink()
    (out_root / ".materialization" / "shards"
     / "toy-00001-of-00001.safetensors.json").unlink()
    stamp_before = shard_path.stat().st_mtime_ns
    rewritten = dsv4_phase8_pack.materialize_shard(
        pack_plan_sha256=plan_sha,
        source=None,
        output_root=out_root,
        source_shard="toy-00001-of-00001.safetensors",
        shard_rows=rows,
        surface=surface,
    )
    # safetensors header metadata serializes in nondeterministic order
    # across processes, so the rewritten file legitimately carries a new
    # shard_sha256; everything semantic must be identical.
    assert rewritten["complete"] is True
    assert rewritten["tensors"] == shard_receipt["tensors"]
    assert rewritten["output_tensor_count"] == (
        shard_receipt["output_tensor_count"])
    assert rewritten["routed_choice_count"] == 2
    assert rewritten["shard_sha256"] == dsv4_phase8_pack.common.sha256_file(
        shard_path)
    assert done.read_text(encoding="utf-8").strip() == (
        rewritten["receipt_sha256"])
    assert shard_path.stat().st_mtime_ns != stamp_before
    print("resume + .done marker discipline OK")

    index = dsv4_phase8_pack.build_output_index([shard_receipt])
    assert set(index["weight_map"]) == {
        record["name"] for record in shard_receipt["tensors"]}
    assert index["metadata"]["total_size"] == expected_bytes
    assert index["metadata"]["total_size"] == (
        shard_receipt["output_logical_bytes"])
    print("index accounting OK (total_size == sum tensor data bytes)")

    print("SELFTEST OK", root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
