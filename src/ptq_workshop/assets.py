"""Pinned source-cache verification shared by preparation and preflight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def snapshot_path(
    cache_dir: str | Path | None, model_id: str, revision: str
) -> Path:
    cache_root = Path(cache_dir or Path.home() / ".cache" / "huggingface" / "hub")
    repository = "models--" + model_id.replace("/", "--")
    return cache_root / repository / "snapshots" / revision


def verify_snapshot(path: Path, model_id: str, revision: str) -> dict[str, Any]:
    """Prove the pinned snapshot and all indexed shards are materialized."""

    snapshot = path.expanduser().resolve()
    if revision not in snapshot.parts:
        raise RuntimeError(f"Snapshot path does not contain pinned revision {revision}: {snapshot}")
    required = ("config.json", "model.safetensors.index.json", "tokenizer.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    incomplete = (
        sorted(snapshot.parents[1].rglob("*.incomplete"))
        if len(snapshot.parents) >= 2
        else []
    )
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
    empty_shards = [
        name
        for name in shard_names
        if (snapshot / name).is_file() and (snapshot / name).stat().st_size == 0
    ]
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
