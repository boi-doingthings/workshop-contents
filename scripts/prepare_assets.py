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

from ptq_workshop.assets import snapshot_path, verify_snapshot
from ptq_workshop.config import (
    CACHED_PREPARATION_MINIMUM_FREE_DISK_GIB,
    SOURCE_DOWNLOAD_MINIMUM_FREE_DISK_GIB,
    get_profile,
    prepared_root_for_profile,
)
from ptq_workshop.datasets import freeze_calibration, freeze_evaluation
from ptq_workshop.io import atomic_write_json
from ptq_workshop.preflight import disk_space_check


def marker_matches(marker: dict[str, Any], profile: Any) -> bool:
    return (
        marker.get("passed") is True
        and marker.get("profile") == profile.name.value
        and marker.get("model_id") == profile.model_id
        and marker.get("model_revision") == profile.model_revision
        and marker.get("storage_budget", {}).get("passed") is True
        and marker.get("storage_budget", {}).get("cache_state") == "download_required"
        and int(marker.get("storage_budget", {}).get("minimum_free_disk_gib", 0))
        == SOURCE_DOWNLOAD_MINIMUM_FREE_DISK_GIB
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="WORKSHOP_B200")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    profile = get_profile(args.profile)
    args.root = args.root or prepared_root_for_profile(Path.cwd(), profile.name)
    args.root.mkdir(parents=True, exist_ok=True)
    marker_path = args.root / "preparation_preflight.json"
    existing_marker = (
        json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.is_file() else None
    )
    expected_snapshot = snapshot_path(
        os.environ.get("HF_HOME"), profile.model_id, profile.model_revision
    )
    try:
        cached_proof = verify_snapshot(expected_snapshot, profile.model_id, profile.model_revision)
    except (RuntimeError, OSError, json.JSONDecodeError):
        cached_proof = None
    cache_state = "verified_complete" if cached_proof is not None else "download_required"
    minimum_free_gib = (
        CACHED_PREPARATION_MINIMUM_FREE_DISK_GIB
        if cached_proof is not None
        else SOURCE_DOWNLOAD_MINIMUM_FREE_DISK_GIB
    )
    preparation_disk = disk_space_check(args.root, minimum_free_gib)
    if not preparation_disk.passed and not (
        cached_proof is None
        and existing_marker is not None
        and marker_matches(existing_marker, profile)
    ):
        raise RuntimeError(
            f"Preparation storage check failed for cache state {cache_state}: "
            f"{preparation_disk.detail}"
        )
    storage_budget = {
        "passed": True,
        "cache_state": cache_state,
        "cached_weight_bytes": 0 if cached_proof is None else cached_proof["weight_bytes"],
        "minimum_free_disk_gib": minimum_free_gib,
        "detail": preparation_disk.detail,
    }
    marker = {
        "passed": True,
        "profile": profile.name.value,
        "model_id": profile.model_id,
        "model_revision": profile.model_revision,
        "storage_budget": storage_budget,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_timing": "after_pinned_cache_inspection",
    }
    atomic_write_json(marker_path, marker)
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
        "storage_budget": storage_budget,
        "calibration": calibration,
        "evaluation": evaluation,
    }
    atomic_write_json(args.root / "prepared_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
