"""Deterministic artifact naming and guarded metadata writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import ProfileName, WorkshopConfig


FINGERPRINT_LENGTH = 16
VARIANTS = ("bf16", "fp8", "nvfp4")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_fingerprint(
    config: WorkshopConfig,
    *,
    recipe_digests: Mapping[str, str] | None = None,
) -> str:
    """Hash canonical experimental inputs, independent of checkout location."""

    payload = config.fingerprint_payload()
    payload["recipes"] = dict(sorted((recipe_digests or {}).items()))
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


def recipe_digests(config: WorkshopConfig) -> dict[str, str]:
    paths = {name: config.recipe_dir / f"{name}.yaml" for name in ("fp8", "nvfp4")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required quantization recipe(s) missing: {', '.join(missing)}")
    return {name: sha256_file(path) for name, path in paths.items()}


@dataclass(frozen=True)
class ArtifactLayout:
    """All outputs for one immutable run fingerprint."""

    artifact_root: Path
    fingerprint: str
    run_name: str | None = None

    def __post_init__(self) -> None:
        if len(self.fingerprint) != FINGERPRINT_LENGTH or any(
            char not in "0123456789abcdef" for char in self.fingerprint
        ):
            raise ValueError(f"Invalid run fingerprint: {self.fingerprint!r}")
        if self.run_name is not None:
            if not self.run_name.startswith("run-") or Path(self.run_name).name != self.run_name:
                raise ValueError(f"Invalid run directory name: {self.run_name!r}")

    @classmethod
    def from_config(
        cls,
        config: WorkshopConfig,
        *,
        now: datetime | None = None,
    ) -> "ArtifactLayout":
        digests = recipe_digests(config)
        fingerprint = run_fingerprint(config, recipe_digests=digests)
        if config.profile.name is ProfileName.DEV_SMOKE:
            run_name = f"run-{fingerprint}"
        else:
            timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
            run_name = f"run-{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}-{fingerprint}"
        return cls(config.artifact_root, fingerprint, run_name)

    @property
    def run_dir(self) -> Path:
        return self.artifact_root / (self.run_name or f"run-{self.fingerprint}")

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def source_dir(self) -> Path:
        """Marker directory for the single pinned BF16 HF snapshot."""

        return self.run_dir / "source"

    def checkpoint_dir(self, variant: str) -> Path:
        self._validate_variant(variant)
        return self.run_dir / "checkpoints" / variant

    def log_path(self, variant: str, stage: str = "quantize") -> Path:
        self._validate_variant(variant)
        if not stage or not stage.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid log stage: {stage!r}")
        return self.run_dir / "logs" / f"{variant}-{stage}.log"

    def metrics_path(self, variant: str, kind: str) -> Path:
        self._validate_variant(variant)
        if not kind or not kind.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid metrics kind: {kind!r}")
        return self.run_dir / "metrics" / f"{variant}-{kind}.json"

    def telemetry_path(self, variant: str) -> Path:
        self._validate_variant(variant)
        return self.run_dir / "telemetry" / f"{variant}.csv"

    def ensure_directories(self) -> None:
        for directory in (
            self.source_dir,
            self.run_dir / "checkpoints",
            self.run_dir / "logs",
            self.run_dir / "metrics",
            self.run_dir / "telemetry",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_variant(variant: str) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown precision {variant!r}; expected one of {VARIANTS}")


def build_manifest(config: WorkshopConfig, layout: ArtifactLayout) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_fingerprint": layout.fingerprint,
        "run_directory": layout.run_dir.name,
        "configuration": config.fingerprint_payload(),
        "recipe_sha256": recipe_digests(config),
    }


def write_json_atomic(path: Path | str, payload: Any, *, overwrite: bool = False) -> Path:
    """Write JSON atomically and refuse unrequested replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if existing == serialized:
            return destination
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite different artifact: {destination}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_derived_json_atomic(path: Path | str, payload: Any) -> Path:
    """Atomically replace a retryable derived metric at its canonical path."""

    return write_json_atomic(path, payload, overwrite=True)


def require_checkpoint_validation(
    layout: ArtifactLayout,
    *,
    source_model: Path | str,
    variants: tuple[str, ...],
) -> dict[str, Any]:
    """Require a matching, still-current checkpoint-validation manifest.

    Direct evaluation and benchmark entry points must consume the durable
    validation gate produced by the ``validate`` stage. The lightweight
    footprint and quant-config checks below reject a stale manifest without
    reopening every tensor in a multi-gigabyte checkpoint.
    """

    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown precision variant(s): {', '.join(unknown)}")
    manifest_path = layout.run_dir / "manifests" / "checkpoint-validation.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Checkpoint validation manifest is required before evaluation or benchmarking: "
            f"{manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Invalid checkpoint validation manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid checkpoint validation manifest: {manifest_path}")

    source = Path(source_model).expanduser().resolve()
    for variant in variants:
        record = manifest.get(variant)
        if not isinstance(record, dict) or record.get("precision") != variant:
            raise ValueError(f"Validation manifest lacks a valid {variant} record")
        expected_path = source if variant == "bf16" else layout.checkpoint_dir(variant).resolve()
        recorded_path = Path(str(record.get("path", ""))).expanduser().resolve()
        if recorded_path != expected_path:
            raise ValueError(
                f"Validation manifest {variant} path mismatch: {recorded_path} != {expected_path}"
            )
        if not expected_path.is_dir():
            raise FileNotFoundError(f"Validated {variant} checkpoint is missing: {expected_path}")
        shards = sorted(expected_path.glob("*.safetensors"))
        current_files = len(shards)
        current_bytes = sum(path.stat().st_size for path in shards)
        if current_files != int(record.get("files", -1)) or current_bytes != int(
            record.get("bytes", -1)
        ):
            raise ValueError(f"Validated {variant} checkpoint footprint has changed")
        if variant != "bf16":
            quant_config = expected_path / "hf_quant_config.json"
            recorded_hash = record.get("hf_quant_config_sha256")
            if not quant_config.is_file() or recorded_hash != sha256_file(quant_config):
                raise ValueError(f"Validated {variant} quantization config has changed")
            normalization = expected_path / ".config_normalization.json"
            recorded_normalization_hash = record.get("config_normalization_sha256")
            if (
                not normalization.is_file()
                or recorded_normalization_hash != sha256_file(normalization)
            ):
                raise ValueError(
                    f"Validated {variant} config-normalization audit has changed"
                )
            quantization_metadata = (
                expected_path / ".quantization_metadata_normalization.json"
            )
            recorded_quantization_metadata_hash = record.get(
                "quantization_metadata_normalization_sha256"
            )
            if (
                not quantization_metadata.is_file()
                or recorded_quantization_metadata_hash
                != sha256_file(quantization_metadata)
            ):
                raise ValueError(
                    f"Validated {variant} quantization-metadata audit has changed"
                )
            source_topology = expected_path / ".source_topology_normalization.json"
            recorded_source_topology_hash = record.get(
                "source_topology_normalization_sha256"
            )
            if (
                not source_topology.is_file()
                or recorded_source_topology_hash != sha256_file(source_topology)
            ):
                raise ValueError(
                    f"Validated {variant} source-topology audit has changed"
                )
    return manifest


def initialize_run(config: WorkshopConfig, layout: ArtifactLayout | None = None) -> ArtifactLayout:
    """Create the layout and an immutable manifest for a reproducible run."""

    resolved = layout or ArtifactLayout.from_config(config)
    expected_fingerprint = run_fingerprint(config, recipe_digests=recipe_digests(config))
    if resolved.fingerprint != expected_fingerprint:
        raise ValueError("Artifact layout fingerprint does not match the configuration and recipes")
    resolved.ensure_directories()
    write_json_atomic(resolved.manifest_path, build_manifest(config, resolved))
    return resolved
