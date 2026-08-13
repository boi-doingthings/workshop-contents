import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ptq_workshop.config import MODEL_ID, MODEL_REVISION


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_assets.py"
    spec = importlib.util.spec_from_file_location("prepare_assets", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _complete_snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "models--nvidia--nemotron" / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["NemotronHForCausalLM"]}), encoding="utf-8"
    )
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    return snapshot


def test_complete_snapshot_verification(tmp_path: Path) -> None:
    module = _module()
    proof = module.verify_snapshot(_complete_snapshot(tmp_path), MODEL_ID, MODEL_REVISION)
    assert proof["verified"] is True
    assert proof["weight_shard_count"] == 1
    assert proof["incomplete_files"] == 0


def test_snapshot_verification_rejects_incomplete_download(tmp_path: Path) -> None:
    module = _module()
    snapshot = _complete_snapshot(tmp_path)
    (snapshot.parents[1] / "blobs").mkdir()
    (snapshot.parents[1] / "blobs" / "pending.incomplete").write_bytes(b"")
    with pytest.raises(RuntimeError, match="incomplete"):
        module.verify_snapshot(snapshot, MODEL_ID, MODEL_REVISION)


def test_marker_must_match_profile_model_and_revision() -> None:
    module = _module()
    profile = SimpleNamespace(
        name=SimpleNamespace(value="DEV_SMOKE"),
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        minimum_free_disk_gib=145,
    )
    marker = {
        "passed": True,
        "profile": "DEV_SMOKE",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "minimum_free_disk_gib": 145,
    }
    assert module.marker_matches(marker, profile)
    assert not module.marker_matches({**marker, "model_revision": "wrong"}, profile)
