#!/usr/bin/env python3
"""Phase 5 probe driver: per-layer K3/K4 probe losses and the sealed
sensitivity-DP allocation for the DSV4 Flash mixed 3.5-bpw campaign.

Port of the GLM-5.3 campaign's k35_probe_driver.py (phase 5) to DSV4
Flash geometry: layers 0..42 all routed (no MTP in encode v1), 256
experts x 3 projections = 768 tensors per layer, floor 3 + 384 K4
upgrades = 2688 bit-units = 3.5 bpw per layer.  The numeric body is
byte-faithful where the adaptation map says the logic is unchanged;
every deviation is marked DSV4 DELTA in a comment.

One layer per invocation (layers 0..42):

  1. open the sealed self-capture (the calibration root points directly
     at the main-full manifest, schema quant-pipeline.dsv4-capture.v1,
     roles fit/conditional-fit/selection/confirmation) and the capped
     dequant-on-demand source,
  2. per 256 experts x 3 projections encode BOTH rates (bits=(3, 4))
     through Exl3MCGCodec.encode_candidates with the pinned campaign
     sigma_reg 0.025,
  3. score each candidate with the campaign's covariance-proxy loss
     (dsv4_common.covariance_proxy_loss),
  4. write probes/L{NN}.json (NEW SURFACE ledger, sealed),
  5. DP-solve the exact 2688-bit-unit / 384-K4 allocation
     (dsv4_common.solve_layer_dp; see the WARN in dsv4_common),
  6. seal non-provisionally (basis="sensitivity_dp_probe_v1") into
     allocations/L{NN}.json under the dsv4 allocation schema,
  7. print exactly one JSON line (the capped source prints its own
     one-line [src] banner first; the JSON line is the last line).

Loss definition (documented choice, unchanged from the GLM probe
driver docstring): the probe loss is the relative covariance quadratic
e^T C e / w^T C w, the loss the sealed K4 numeric closure computes in
Exl3TrellisCodec.encode (r7_encoder/trellis.py:383-396), recomputed
over the R10 candidate reconstruction.  The R10 path returns
proxy_loss=0.0 by design (r10_codec.py:512), so the driver evaluates
the identical formula; R10 documents byte-compatibility with the R7
audited path, and the bridge is recorded in the ledger header.
Optional cross-check (not run here): encode one tensor through the R7
audited path and compare proxy_loss directly.

Down conditioning (documented choice, unchanged): the down-projection
curve is measured under ONE conditioning context per expert, the R7
pair_at semantics (r7_encoder/layer.py:901-925 memoizes one context
per gate/up rate pair; layer.py:974-994 runs every down bit width
under pair_at(base_gate_bits, base_up_bits)).  Both candidate rates
(3 and 4) are conditioned on w1/w3 decoded at the reference rates
dsv4_common.FLOOR_BITS, so loss@3 and loss@4 are same-denominator
relative quadratics and the DP gain mass*(loss3-loss4) subtracts one
metric.  The encode worker conditions its encode-time down Hessian on
the identical context.  This is the fixed_point_iteration=0 probe; the
lineage's iterate-to-fixed-point refinement (r7_encoder/sensitivity.py
ProbeLedger) is not ported here.

Probe-time vectors (DSV4 DELTA): the GLM probe fed K4-rate GSS
preparation vectors to both rates (GLM probe driver lines 42-44,
147-149, 213-215) because a sealed uniform-K4 campaign already
existed.  No DSV4 GSS exists at probe time (phase 6 builds the
rate-specific GSS after this phase), so this probe encodes under the
neutral uniform fp16 ones vector, identical for both candidate rates
and all projections; loss@3 and loss@4 therefore stay
same-denominator and the DP gain subtracts one metric.  The binding is
recorded in the ledger header; the final encode uses rate-specific GSS
vectors.

Weights source (DSV4 DELTA): dsv4_capped_source.CappedSource
dequantizes the packed master on demand; load_expert returns
(w1, w2, w3) bf16 in engine order (gate, down, up), and expert tensors
carry no LoRA sites, so this is pure dequant.  The GLM probe applied a
prior-search permutation to each triplet (GLM probe driver lines
131-135); DSV4 has no prior-search permutation at probe time and
encodes the logical weight directly (identity permutation).

Hash-routed layers 0..2 (DSV4): routing there is token-id
deterministic, and a narrow corpus can leave experts with zero fit
rows; the probe fail-closes inside dsv4_common.expert_p2_mass.  The
capture corpus must be genuinely broad before this driver is run.

Usage (inside the encode container, cwd /wd, engine files present,
PYTHONPATH carrying the quant_pipeline campaign repo and the r10
bundle):

  python3 dsv4_probe_driver.py --layer 0 --work-root /workspace

ASCII only.  No em-dashes.  No network.  Writes only under --work-root.
CODE ONLY: written for the encode pod; nothing here is executed at
authoring time.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsv4_common as common
from dsv4_common import die

# Codec-boundary vocabulary: the ONE place master w-names map to the
# adapter's projection names.  Exl3MCGCodec._parse_unit only accepts
# gate_proj/up_proj/down_proj inside unit_id (codecs/exl3_mcg.py:153-155)
# and silently degrades a non-matching unit_id to a placeholder identity,
# so this translation is load-bearing.  Master names (w1/w2/w3) rule
# everywhere else: ledger keys, allocation keys, provenance, DP census.
CODEC_PROJECTION = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}


def codec_projection(projection: str) -> str:
    """Translate a master projection name for the codec unit_id ONLY."""
    if projection not in CODEC_PROJECTION:
        die(f"unknown master projection {projection!r}")
    return CODEC_PROJECTION[projection]


# NEW SURFACE: the GLM campaign sealed allocations through its uniform-k35
# module (called by the GLM probe driver lines 322-325); DSV4 has no such
# module, so this driver seals the same receipt shape under a dsv4 schema
# (invented here per the port contract).
DSV4_ALLOCATION_SCHEMA = "quant-pipeline.dsv4-k35-layer-allocation.v1"


def parse_args() -> argparse.Namespace:
    """Mirror of the GLM probe driver's parse_args (k35_probe_driver.py
    lines 69-86) with the DSV4 arg surface.  DSV4 DELTA: dsv4_common has
    no add_common_args/finish_common_args, so the plumbing lives here;
    the BF16-fold args (--bf16-root, --verify-shards) and the
    preparation root are gone (no fold, no GSS preparation at probe
    time); the capped-source paths are args instead."""
    parser = argparse.ArgumentParser(description="phase 5 dsv4 probe driver")
    parser.add_argument(
        "--layer", required=True, type=int, help="0..42 (all routed; no MTP in encode v1)"
    )
    parser.add_argument("--work-root", default=str(common.DEFAULT_WORK_ROOT), type=Path)
    parser.add_argument(
        "--calibration-root",
        default=str(common.DEFAULT_CALIBRATION_ROOT),
        type=Path,
        help="capture manifest root (default .../calibration/main-full)",
    )
    parser.add_argument(
        "--model-dir", default=str(common.MODEL_DIR),
        help="packed DSV4 master (default /model or env DSV4_MODEL)",
    )
    parser.add_argument(
        "--meta-path", default=str(common.META_PATH),
        help="tensor meta (default /wd/tensor_meta.json or env DSV4_META)",
    )
    parser.add_argument(
        "--lora-path", default=str(common.LORA_PATH),
        help="cap LoRA (default /wd/cap/lora.safetensors or env DSV4_LORA)",
    )
    parser.add_argument("--device", default=None, help="default: cuda:0")
    parser.add_argument(
        "--extension",
        default=None,
        help=f"path to the compiled exllamav3_ext .so (or env {common.ENV_EXTENSION})",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="r10 bundle root containing r7_encoder/ (default: discovered on sys.path)",
    )
    parser.add_argument("--chunk-rows", default=common.CHUNK_ROWS, type=int)
    parser.add_argument(
        "--no-verify-capture-hashes",
        action="store_true",
        help="skip per-layer capture payload hashing (manifest seal still verified)",
    )
    args = parser.parse_args()
    args.work_root = Path(args.work_root).resolve()
    args.calibration_root = Path(args.calibration_root).resolve()
    if args.device is None:
        args.device = "cuda:0"
    if args.layer not in common.ALL_PROBE_LAYERS:
        die(f"--layer must be one of {list(common.MAIN_LAYERS)}")
    return args


def capture_binding(capture) -> dict:
    """Ledger binding for the sealed capture.

    DSV4 DELTA: the GLM capture view exposed binding() with the GLM
    campaign's inventory and token-panel receipt fields; dsv4_common's
    Dsv4CaptureView keeps no manifest reference, so re-read the sealed
    manifest next to the view.  Only fields the dsv4 capture schema
    actually carries are bound; no GLM receipt fields are invented."""
    manifest = common.load_json(Path(capture.root) / "capture-manifest.json")
    rows = int(capture.ids.shape[0])
    if manifest.get("rows_per_layer") != rows:
        die("capture manifest row census disagrees with the memmap")
    return {
        "capture_sha256": common.require_hash(
            manifest.get("capture_sha256"), "capture seal"
        ),
        "layer": int(capture.layer),
        "roles": list(manifest["roles"]),
        "rows": rows,
        "producer": manifest["producer"],
        "route_id_abi": "uint16-little-endian",
    }


def check_expert_shape(layer: int, expert: int, projection: str, weight) -> None:
    """Fail-closed orientation gate on the dequanted logical tensor
    (HF [N,K], engine computes x @ w.T).  A transposed or mis-shaped
    source would otherwise flow through every seal green; the expected
    shapes are re-derived from dsv4_common constants."""
    import torch

    expected = {
        "w1": (common.INTERMEDIATE_SIZE, common.HIDDEN_SIZE),
        "w3": (common.INTERMEDIATE_SIZE, common.HIDDEN_SIZE),
        "w2": (common.HIDDEN_SIZE, common.INTERMEDIATE_SIZE),
    }
    shape = tuple(int(v) for v in torch.as_tensor(weight).shape)
    if projection not in expected:
        die(f"unknown master projection {projection!r}")
    if shape != expected[projection]:
        die(
            f"L{layer} E{expert} {projection}: logical shape {shape} differs "
            f"from {expected[projection]} (orientation or geometry drift)"
        )


def uniform_probe_vectors(weight, device: str):
    """Neutral fp16 ones (suh over the input dim K, svh over the output
    dim N) for the probe encode.  DSV4 DELTA: replaces the GLM K4-rate
    GSS preparation vectors (GLM probe driver lines 147-149, 213-215);
    see the module docstring's Probe-time vectors paragraph."""
    import torch

    n, k = (int(v) for v in torch.as_tensor(weight).shape)
    return (
        torch.ones(k, dtype=torch.float16, device=device),
        torch.ones(n, dtype=torch.float16, device=device),
    )


def seal_layer_allocation(layer: int, allocation: dict) -> dict:
    """Non-provisional allocation receipt under the dsv4 schema.

    Mirror of the GLM campaign's seal_layer_allocation (uniform-k35
    module lines 380-394, invoked by the GLM probe driver lines
    322-325): same receipt fields, DSV4 census constants, and the 7/2
    rate re-derived fail-closed from dsv4_common instead of copied."""
    common.audit_layer_allocation(layer, allocation)
    if common.TARGET_BIT_UNITS_PER_LAYER * 2 != common.TENSORS_PER_LAYER * 7:
        die("3.5-bpw rate arithmetic does not close at the DSV4 census")
    body = {
        "schema": DSV4_ALLOCATION_SCHEMA,
        "layer": layer,
        "allocation": dict(sorted(allocation.items())),
        "bit_units": common.TARGET_BIT_UNITS_PER_LAYER,
        "k4_tensor_count": common.K4_TENSORS_PER_LAYER,
        "k3_tensor_count": common.K3_TENSORS_PER_LAYER,
        "rate": {"numerator": 7, "denominator": 2},
        "provisional": False,
        "basis": "sensitivity_dp_probe_v1",
    }
    return common.seal(body, "allocation_sha256")


def role_row_counts(capture, expert: int) -> dict[str, int]:
    """Byte-faithful mirror of the GLM probe driver lines 89-93."""
    return {
        role: int(capture.routed_rows(expert, role).rows)
        for role in ("fit", "conditional-fit", "selection", "confirmation")
    }


def candidate_record(weight_hf, candidate, covariance) -> dict:
    """Byte-faithful mirror of the GLM probe driver lines 96-103."""
    loss = common.covariance_proxy_loss(weight_hf, candidate.reconstructed, covariance)
    return {
        "loss": format(loss, ".17g"),
        "packed_sha256": candidate.packed_sha256,
        "reconstruction_sha256": candidate.reconstruction_sha256,
        "stored_bytes": int(candidate.stored_bytes),
    }


def main() -> None:
    args = parse_args()
    layer = args.layer

    capture = common.open_capture(
        args.calibration_root, layer, verify_hashes=not args.no_verify_capture_hashes
    )
    binding = capture_binding(capture)

    # Weights source: the capped dequant-on-demand source (DSV4 DELTA:
    # replaces the GLM sealed BF16 fold source).  Imported here because
    # full_loader is a pod engine file at cwd; the module imports no
    # r7_encoder code, so the codec-first import order below is safe.
    from dsv4_capped_source import CappedSource

    source = CappedSource(
        model_dir=args.model_dir,
        meta_path=args.meta_path,
        lora_path=args.lora_path,
        device=args.device,
    )

    # Codec constructed BEFORE any r7_encoder import (the sealed import
    # order dsv4_common.build_codec enforces, mirroring the GLM common
    # module's build_codec contract).
    source_root = common.resolve_source_root(args)
    codec = common.build_codec(source_root, common.resolve_extension(args), args.device)
    codec_identity_sha256 = common.require_hash(
        common.sha256_bytes(common.canonical_json(codec.identity)), "codec identity"
    )

    records = []
    masses = []
    for expert in range(common.NUM_EXPERTS):
        # Engine order (gate, down, up) -> master roles w1, w2, w3.
        # DSV4 DELTA vs the GLM probe driver lines 128-135: no sealed
        # BF16 triplet dict and no prior-search permutation; the logical
        # dequanted weight encodes directly, gated by a fail-closed
        # shape check per projection.
        w1, w2, w3 = source.load_expert(layer, expert)
        gate_weight = w1.to(args.device)
        down_weight = w2.to(args.device)
        up_weight = w3.to(args.device)
        for projection, weight in (
            ("w1", gate_weight),
            ("w2", down_weight),
            ("w3", up_weight),
        ):
            check_expert_shape(layer, expert, projection, weight)

        gate_cov, gate_cov_evidence = common.gate_covariance(
            codec, capture, expert, args.device, args.chunk_rows
        )
        gate_cov_evidence["matrix_sha256"] = common.tensor_sha256(gate_cov)

        # w1/w3: encode BOTH rates per tensor, keeping the candidate
        # objects alive for the down conditioning decode (GLM probe
        # driver lines 143-171, keys renamed to master w-names).
        encoded_gate_up = {}
        projections_record = {}
        for projection, weight in (("w1", gate_weight), ("w3", up_weight)):
            suh, svh = uniform_probe_vectors(weight, args.device)
            candidates = codec.encode_candidates(
                # codec-boundary ONLY: the adapter's unit_id parser
                # requires its own projection vocabulary
                # (codecs/exl3_mcg.py:153-155)
                unit_id=f"L{layer}.E{expert}.{codec_projection(projection)}",
                weight_hf=weight,
                covariance=gate_cov,
                bits=(3, 4),
                input_vector=suh,
                output_vector=svh,
                provenance={
                    "dsv4_probe": True,
                    "layer": layer,
                    "expert": expert,
                    "projection": projection,
                },
            )
            encoded_gate_up[projection] = candidates
            projections_record[projection] = {
                str(bits): candidate_record(weight, candidates[bits], gate_cov)
                for bits in (3, 4)
            }
            projections_record[projection]["covariance_matrix_sha256"] = (
                gate_cov_evidence["matrix_sha256"]
            )

        # Corrected operation order: a fresh factor domain for the
        # candidate-conditioned down covariance after exact w1/w3 decode
        # (mirror of the GLM probe driver lines 173-176, itself citing
        # the GLM prepared backend lines 490-492).
        codec._codec().clear_caches()

        # ONE conditioning context for the whole down curve (R7 pair_at
        # semantics, r7_encoder/layer.py:901-925 and 974-994): w1/w3
        # decoded at the reference rates dsv4_common.FLOOR_BITS for BOTH
        # candidate rates, so loss@3 and loss@4 share one Hessian and one
        # denominator and the DP gain mass*(loss3-loss4) subtracts
        # same-metric losses.  The encode worker conditions its
        # encode-time Hessian on this exact context.
        reference_bits = common.FLOOR_BITS
        gate_reference = encoded_gate_up["w1"][reference_bits]
        up_reference = encoded_gate_up["w3"][reference_bits]
        # Candidate reconstructions are HF [N,K]
        # (codecs/exl3_mcg.py:196); down_inputs_from_roundtrip wants
        # [K,N], matching the GLM backend's direct pass of
        # reconstructed_kn (GLM prepared backend lines 284-289).
        down_cov, down_cov_evidence = common.down_covariance(
            codec,
            capture,
            expert,
            gate_reference.reconstructed.t().contiguous(),
            up_reference.reconstructed.t().contiguous(),
            gate_bits=reference_bits,
            up_bits=reference_bits,
            device=args.device,
            chunk_rows=args.chunk_rows,
        )
        down_cov_evidence["matrix_sha256"] = common.tensor_sha256(down_cov)
        down_record = {
            "covariance_matrix_sha256": down_cov_evidence["matrix_sha256"],
            "covariance_evidence": down_cov_evidence,
            "conditioning": {
                "gate_bits": reference_bits,
                "up_bits": reference_bits,
                "semantics": "r7_pair_at_reference_rates_v1",
            },
        }
        for rate in (3, 4):
            suh, svh = uniform_probe_vectors(down_weight, args.device)
            candidates = codec.encode_candidates(
                # codec-boundary ONLY (codecs/exl3_mcg.py:153-155)
                unit_id=f"L{layer}.E{expert}.{codec_projection('w2')}",
                weight_hf=down_weight,
                covariance=down_cov,
                bits=(rate,),
                input_vector=suh,
                output_vector=svh,
                provenance={
                    "dsv4_probe": True,
                    "layer": layer,
                    "expert": expert,
                    "projection": "w2",
                    "conditioning_gate_bits": reference_bits,
                    "conditioning_up_bits": reference_bits,
                },
            )
            entry = candidate_record(down_weight, candidates[rate], down_cov)
            entry["gate_up_roundtrip_sha256"] = {
                "gate": gate_reference.reconstruction_sha256,
                "up": up_reference.reconstruction_sha256,
            }
            down_record[str(rate)] = entry
            del candidates
        projections_record["w2"] = down_record

        mass = common.expert_p2_mass(capture, expert)
        masses.append(Decimal(format(mass, ".17g")))
        records.append(
            {
                "expert": expert,
                "tensor_names": {
                    projection: common.tensor_full_name(layer, expert, projection)
                    for projection in common.PROJECTIONS
                },
                "p2_mass": format(mass, ".17g"),
                "row_counts": role_row_counts(capture, expert),
                "gate_up_covariance": gate_cov_evidence,
                "projections": projections_record,
            }
        )
        del (
            w1,
            w2,
            w3,
            gate_weight,
            up_weight,
            down_weight,
            gate_cov,
            down_cov,
            encoded_gate_up,
        )
        if expert % 16 == 15:
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Ledger header mirrors the GLM probe driver lines 272-310 with two
    # DSV4 DELTAs: the GLM inventory_sha256 binding is replaced by the
    # capped-source identity (no BF16 fold inventory exists), and the
    # GLM preparation_binding block is dropped in favor of the honest
    # probe_vectors text (no GSS preparation exists at probe time).
    ledger = common.seal(
        {
            "schema": common.DSV4_PROBE_LEDGER_SCHEMA,
            "layer": layer,
            "capture_binding": binding,
            "source_binding": {
                "source": "dsv4_capped_source.CappedSource",
                "model_dir": str(Path(args.model_dir)),
                "meta_path": str(Path(args.meta_path)),
                "lora_sha256": source.identity["lora_sha256"],
                "lora_scale": source.identity["lora_scale"],
                "lora_sites": source.identity["lora_sites"],
            },
            "codec_identity_sha256": codec_identity_sha256,
            "sigma_reg": common.SIGMA_REG,
            "loss_definition": (
                "relative covariance quadratic e^T C e / w^T C w; formula verbatim "
                "from r7_encoder/trellis.py:383-396 (Exl3TrellisCodec.encode), "
                "recomputed over the R10 candidate reconstruction "
                "(r10_codec.py:512 returns proxy_loss=0.0 by design); R10/R7 "
                "byte-compatibility is documented in the r10_codec.py module "
                "docstring"
            ),
            "probe_loss_bridge": "r10_candidate_reconstruction_plus_r7_quadratic_v1",
            "down_conditioning": (
                "conditional-fit Hessian conditioned on w1/w3 decoded at the "
                "reference rates k3/k3 (dsv4_common.FLOOR_BITS) for BOTH "
                "candidate rates; R7 pair_at semantics (r7_encoder/layer.py:"
                "901-925, 974-994); the DP gain mass*(loss3-loss4) subtracts "
                "same-denominator ratios; the encode worker uses the identical "
                "context; fixed_point_iteration=0"
            ),
            "fixed_point_iteration": 0,
            "probe_vectors": (
                "uniform fp16 ones for both rates and all projections; no GSS "
                "preparation exists at probe time (phase 6 builds the "
                "rate-specific GSS after this phase); both candidate rates "
                "share identical vectors, so loss@3 and loss@4 remain "
                "same-denominator; the final encode uses rate-specific GSS "
                "vectors"
            ),
            "records": records,
        },
        "probe_sha256",
    )

    # DSV4 DELTA (mechanical): dsv4_common.write_json does not create
    # parent directories (the campaign's atomic writer did), so create
    # the two artifact roots explicitly.
    probes_dir = args.work_root / "probes"
    allocations_dir = args.work_root / "allocations"
    probes_dir.mkdir(parents=True, exist_ok=True)
    allocations_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(probes_dir / f"{common.probe_stem(layer)}.json", ledger)

    loss_by_bits = {}
    for record in records:
        for projection in common.PROJECTIONS:
            name = record["tensor_names"][projection]
            loss3 = Decimal(record["projections"][projection]["3"]["loss"])
            loss4 = Decimal(record["projections"][projection]["4"]["loss"])
            loss_by_bits[name] = (loss3, loss4)
    allocation_bits = common.solve_layer_dp(layer, loss_by_bits, masses)
    allocation = seal_layer_allocation(layer, allocation_bits)
    common.write_json(
        allocations_dir / f"{common.probe_stem(layer)}.json", allocation
    )

    worst_k3_loss = max(
        float(record["projections"][projection]["3"]["loss"])
        for record in records
        for projection in common.PROJECTIONS
        if allocation_bits[record["tensor_names"][projection]] == 3
    )
    print(
        json.dumps(
            {
                "layer": layer,
                "k4_tensors": allocation["k4_tensor_count"],
                "worst_k3_loss": worst_k3_loss,
                "alloc_sha256": allocation["allocation_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
