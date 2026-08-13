from datetime import datetime, timezone
from pathlib import Path

import pytest

from ptq_workshop.artifacts import (
    ArtifactLayout,
    initialize_run,
    recipe_digests,
    require_checkpoint_validation,
    run_fingerprint,
    sha256_file,
    write_derived_json_atomic,
    write_json_atomic,
)
from ptq_workshop.config import make_config


def _recipes(root: Path) -> None:
    directory = root / "configs" / "recipes"
    directory.mkdir(parents=True)
    (directory / "fp8.yaml").write_text("fp8\n", encoding="utf-8")
    (directory / "nvfp4.yaml").write_text("nvfp4\n", encoding="utf-8")


def test_fingerprint_is_path_independent(tmp_path: Path) -> None:
    roots = (tmp_path / "one", tmp_path / "two")
    for root in roots:
        _recipes(root)
    first = make_config("DEV_SMOKE", project_root=roots[0])
    second = make_config("DEV_SMOKE", project_root=roots[1])
    assert run_fingerprint(first, recipe_digests=recipe_digests(first)) == run_fingerprint(
        second, recipe_digests=recipe_digests(second)
    )


def test_dev_resumes_but_workshop_is_timestamped(tmp_path: Path) -> None:
    _recipes(tmp_path)
    instant = datetime(2026, 8, 12, 5, 30, tzinfo=timezone.utc)
    dev = ArtifactLayout.from_config(make_config("DEV_SMOKE", project_root=tmp_path), now=instant)
    workshop = ArtifactLayout.from_config(
        make_config("WORKSHOP_B200", project_root=tmp_path), now=instant
    )
    assert dev.run_dir.name == f"run-{dev.fingerprint}"
    assert workshop.run_dir.name.startswith("run-20260812T053000.000000Z-")
    assert workshop.run_dir.name.endswith(workshop.fingerprint)


def test_initialize_writes_immutable_manifest(tmp_path: Path) -> None:
    _recipes(tmp_path)
    config = make_config("DEV_SMOKE", project_root=tmp_path)
    layout = initialize_run(config)
    assert layout.manifest_path.is_file()
    assert initialize_run(config).run_dir == layout.run_dir
    with pytest.raises(FileExistsError):
        write_json_atomic(layout.manifest_path, {"different": True})


def test_retryable_derived_json_write_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "metrics" / "energy.json"
    write_derived_json_atomic(path, {"energy_j": 10})
    write_derived_json_atomic(path, {"energy_j": 12})
    assert path.read_text(encoding="utf-8") == '{\n  "energy_j": 12\n}\n'


def test_artifact_names_accept_hyphen_and_underscore_separators(tmp_path: Path) -> None:
    layout = ArtifactLayout(tmp_path, "0" * 16)
    assert layout.log_path("fp8", "benchmark-serve").name == "fp8-benchmark-serve.log"
    assert layout.metrics_path("fp8", "runtime_status").name == "fp8-runtime_status.json"
    for invalid in ("../serve", "serve.log", "serve space", ""):
        with pytest.raises(ValueError):
            layout.log_path("fp8", invalid)


def _validated_layout(tmp_path: Path) -> tuple[ArtifactLayout, Path]:
    layout = ArtifactLayout(tmp_path / "runs", "a" * 16, f"run-{'a' * 16}")
    source = tmp_path / "source"
    source.mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"bf16")
    fp8 = layout.checkpoint_dir("fp8")
    fp8.mkdir(parents=True)
    (fp8 / "model.safetensors").write_bytes(b"fp8")
    (fp8 / "hf_quant_config.json").write_text('{"quantization": "FP8"}\n')
    (fp8 / ".config_normalization.json").write_text('{"after_sha256": "fixture"}\n')
    (fp8 / ".quantization_metadata_normalization.json").write_text(
        '{"config_after_sha256": "fixture"}\n'
    )
    (fp8 / ".source_topology_normalization.json").write_text(
        '{"config_after_sha256": "fixture"}\n'
    )
    payload = {
        "bf16": {
            "precision": "bf16",
            "path": str(source.resolve()),
            "files": 1,
            "bytes": 4,
        },
        "fp8": {
            "precision": "fp8",
            "path": str(fp8.resolve()),
            "files": 1,
            "bytes": 3,
            "hf_quant_config_sha256": sha256_file(fp8 / "hf_quant_config.json"),
            "config_normalization_sha256": sha256_file(
                fp8 / ".config_normalization.json"
            ),
            "quantization_metadata_normalization_sha256": sha256_file(
                fp8 / ".quantization_metadata_normalization.json"
            ),
            "source_topology_normalization_sha256": sha256_file(
                fp8 / ".source_topology_normalization.json"
            ),
        },
    }
    write_json_atomic(
        layout.run_dir / "manifests" / "checkpoint-validation.json",
        payload,
    )
    return layout, source


def test_runtime_requires_matching_checkpoint_validation_manifest(tmp_path: Path) -> None:
    layout, source = _validated_layout(tmp_path)
    manifest = require_checkpoint_validation(
        layout,
        source_model=source,
        variants=("bf16", "fp8"),
    )
    assert set(manifest) == {"bf16", "fp8"}


def test_runtime_rejects_missing_or_stale_checkpoint_validation(tmp_path: Path) -> None:
    layout = ArtifactLayout(tmp_path / "missing", "b" * 16, f"run-{'b' * 16}")
    with pytest.raises(FileNotFoundError, match="validation manifest is required"):
        require_checkpoint_validation(layout, source_model=tmp_path, variants=("bf16",))

    layout, source = _validated_layout(tmp_path / "stale")
    (layout.checkpoint_dir("fp8") / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="footprint has changed"):
        require_checkpoint_validation(layout, source_model=source, variants=("fp8",))


def test_runtime_rejects_stale_config_normalization_audit(tmp_path: Path) -> None:
    layout, source = _validated_layout(tmp_path)
    (layout.checkpoint_dir("fp8") / ".config_normalization.json").write_text(
        '{"after_sha256": "changed"}\n'
    )
    with pytest.raises(ValueError, match="config-normalization audit has changed"):
        require_checkpoint_validation(layout, source_model=source, variants=("fp8",))


def test_runtime_rejects_stale_quantization_metadata_audit(tmp_path: Path) -> None:
    layout, source = _validated_layout(tmp_path)
    (layout.checkpoint_dir("fp8") / ".quantization_metadata_normalization.json").write_text(
        '{"config_after_sha256": "changed"}\n'
    )
    with pytest.raises(ValueError, match="quantization-metadata audit has changed"):
        require_checkpoint_validation(layout, source_model=source, variants=("fp8",))


def test_runtime_rejects_stale_source_topology_audit(tmp_path: Path) -> None:
    layout, source = _validated_layout(tmp_path)
    (layout.checkpoint_dir("fp8") / ".source_topology_normalization.json").write_text(
        '{"config_after_sha256": "changed"}\n'
    )
    with pytest.raises(ValueError, match="source-topology audit has changed"):
        require_checkpoint_validation(layout, source_model=source, variants=("fp8",))
