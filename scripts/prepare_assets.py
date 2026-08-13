#!/usr/bin/env python3
"""Download only the pinned BF16 source and freeze public datasets."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from ptq_workshop.config import POST_PREP_MINIMUM_FREE_DISK_GIB, get_profile
from ptq_workshop.datasets import freeze_calibration, freeze_evaluation
from ptq_workshop.io import atomic_write_json
from ptq_workshop.preflight import disk_space_check


def _snapshot_path(cache_dir: str | None, model_id: str, revision: str) -> Path:
    cache_root = Path(cache_dir or Path.home() / ".cache" / "huggingface" / "hub")
    repository = "models--" + model_id.replace("/", "--")
    return cache_root / repository / "snapshots" / revision


def verify_snapshot(path: Path, model_id: str, revision: str) -> dict[str, Any]:
    """Prove the pinned snapshot and all indexed shards are fully materialized."""

    snapshot = path.expanduser().resolve()
    if revision not in snapshot.parts:
        raise RuntimeError(f"Snapshot path does not contain pinned revision {revision}: {snapshot}")
    required = ("config.json", "model.safetensors.index.json", "tokenizer.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    incomplete = sorted(snapshot.parents[1].rglob("*.incomplete")) if len(snapshot.parents) >= 2 else []
    broken_links = [str(item) for item in snapshot.rglob("*") if item.is_symlink() and not item.exists()]
    if missing or incomplete or broken_links:
        raise RuntimeError(
            f"Pinned snapshot is incomplete: missing={missing}, incomplete={len(incomplete)}, "
            f"broken_links={len(broken_links)}"
        )
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise RuntimeError(f"Snapshot index has no weight shards: {snapshot}")
    absent_shards = [name for name in shard_names if not (snapshot / name).is_file()]
    empty_shards = [name for name in shard_names if (snapshot / name).is_file() and (snapshot / name).stat().st_size == 0]
    if absent_shards or empty_shards:
        raise RuntimeError(
            f"Snapshot weight shards are incomplete: absent={absent_shards}, empty={empty_shards}"
        )
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["NemotronHForCausalLM"]:
        raise RuntimeError(f"Unexpected model architecture in {snapshot / 'config.json'}")
    return {
        "verified": True,
        "model_id": model_id,
        "model_revision": revision,
        "snapshot": str(snapshot),
        "weight_shard_count": len(shard_names),
        "weight_bytes": sum((snapshot / name).stat().st_size for name in shard_names),
        "incomplete_files": 0,
        "broken_links": 0,
    }


def marker_matches(marker: dict[str, Any], profile: Any) -> bool:
    return (
        marker.get("passed") is True
        and marker.get("profile") == profile.name.value
        and marker.get("model_id") == profile.model_id
        and marker.get("model_revision") == profile.model_revision
        and int(marker.get("minimum_free_disk_gib", 0)) == profile.minimum_free_disk_gib
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="WORKSHOP_B200")
    parser.add_argument("--root", type=Path, default=Path("artifacts/prepared"))
    args = parser.parse_args()
    profile = get_profile(args.profile)
    args.root.mkdir(parents=True, exist_ok=True)
    marker_path = args.root / "preparation_preflight.json"
    existing_marker = (
        json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.is_file() else None
    )
    preparation_disk = disk_space_check(args.root, profile.minimum_free_disk_gib)
    expected_snapshot = _snapshot_path(os.environ.get("HF_HOME"), profile.model_id, profile.model_revision)
    recovery: dict[str, Any] | None = None
    if preparation_disk.passed:
        marker = {
            "passed": True,
            "profile": profile.name.value,
            "model_id": profile.model_id,
            "model_revision": profile.model_revision,
            "minimum_free_disk_gib": profile.minimum_free_disk_gib,
            "detail": preparation_disk.detail,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "gate_timing": "before_snapshot_download",
        }
        atomic_write_json(marker_path, marker)
    elif existing_marker is not None and marker_matches(existing_marker, profile):
        marker = existing_marker
        # A persisted pre-transfer marker makes an interrupted snapshot_download
        # resumable without incorrectly requiring the original free space again.
    else:
        postprep_disk = disk_space_check(args.root, POST_PREP_MINIMUM_FREE_DISK_GIB)
        if not postprep_disk.passed:
            raise RuntimeError(
                "Preparation lacks a valid 145 GiB pre-download marker and the verified-cache "
                f"recovery budget failed: {postprep_disk.detail}"
            )
        snapshot_proof = verify_snapshot(
            expected_snapshot, profile.model_id, profile.model_revision
        )
        marker = {
            "passed": False,
            "profile": profile.name.value,
            "model_id": profile.model_id,
            "model_revision": profile.model_revision,
            "minimum_free_disk_gib": profile.minimum_free_disk_gib,
            "gate_timing": "not_recorded",
        }
        recovery = {
            "recovered_complete_cache": True,
            "historical_145_gib_gate_recorded": False,
            "remaining_artifact_budget_gib": POST_PREP_MINIMUM_FREE_DISK_GIB,
            "remaining_artifact_budget_detail": postprep_disk.detail,
            "snapshot_verification": snapshot_proof,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    model_path = snapshot_download(
        repo_id=profile.model_id,
        revision=profile.model_revision,
        cache_dir=os.environ.get("HF_HOME"),
    )
    snapshot_verification = verify_snapshot(
        Path(model_path), profile.model_id, profile.model_revision
    )
    calibration = freeze_calibration(args.root / "calibration.jsonl", profile.calibration_samples, profile.seed)
    evaluation = freeze_evaluation(
        args.root / "evaluation",
        profile.mmlu_samples,
        profile.gsm8k_samples,
        profile.seed,
    )
    manifest = {
        "download_only": True,
        "profile": profile.name,
        "model_id": profile.model_id,
        "model_revision": profile.model_revision,
        "model_snapshot": model_path,
        "preparation_preflight": marker,
        "snapshot_verification": snapshot_verification,
        "recovery": recovery,
        "calibration": calibration,
        "evaluation": evaluation,
    }
    atomic_write_json(args.root / "prepared_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
