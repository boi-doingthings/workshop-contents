"""Immutable experiment configuration for the Blackwell PTQ workshop.

The source checkpoint and Model Optimizer release are intentionally constants,
not user-selectable defaults.  A different model or revision is a different
experiment and should be introduced as an explicit code change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "2d59de1cbd51c0adf384eb906b766d1aee0e0517"
# Controlled compatibility exception: ModelOpt 0.45.0 can calibrate NemotronH
# experts, but its release branch intentionally omitted unified-HF export for
# that non-gated fused-MoE layout. 0.46.0rc0 is the first official NVIDIA tag
# containing the complete quantize + export fix.
MODELOPT_VERSION = "0.46.0rc0"
MODELOPT_SOURCE_TAG = "0.46.0rc0"
MODELOPT_GIT_COMMIT = "33d05b0c446f528914173041057050f6d135fbf4"
MODELOPT_NEMOTRON_EXPORT_FIX_COMMIT = "c81210faecc096a7bd802cca2cda909ac43f7759"
MODELOPT_EXAMPLE = Path("examples/hf_ptq/hf_ptq.py")
TRTLLM_VERSION = "1.3.0rc17"
TRTLLM_IMAGE = (
    "nvcr.io/nvidia/tensorrt-llm/release@"
    "sha256:998068efffcddb06905b83e9e712a4aec9f39d8f1ec4afacf6c0f3bac4479b54"
)
CALIBRATION_DATASET = "local-frozen-cnn-dailymail-jsonl"
SEED = 42
POST_PREP_MINIMUM_FREE_DISK_GIB = 55


class ProfileName(str, Enum):
    """Named workload sizes used throughout the notebook and scripts."""

    DEV_SMOKE = "DEV_SMOKE"
    WORKSHOP_B200 = "WORKSHOP_B200"
    FULL = "FULL"


@dataclass(frozen=True)
class Profile:
    name: ProfileName
    calibration_samples: int
    calibration_sequence_length: int
    calibration_batch_size: int
    mmlu_samples: int
    gsm8k_samples: int
    benchmark_requests: int
    benchmark_warmup_requests: int
    benchmark_repetitions: int
    minimum_free_disk_gib: int

    @property
    def model_id(self) -> str:
        return MODEL_ID

    @property
    def model_revision(self) -> str:
        return MODEL_REVISION

    @property
    def seed(self) -> int:
        return SEED

    @property
    def evaluation_samples(self) -> int:
        return self.mmlu_samples + self.gsm8k_samples

    def __post_init__(self) -> None:
        positive = {
            "calibration_samples": self.calibration_samples,
            "calibration_sequence_length": self.calibration_sequence_length,
            "calibration_batch_size": self.calibration_batch_size,
            "mmlu_samples": self.mmlu_samples,
            "gsm8k_samples": self.gsm8k_samples,
            "benchmark_requests": self.benchmark_requests,
            "benchmark_repetitions": self.benchmark_repetitions,
            "minimum_free_disk_gib": self.minimum_free_disk_gib,
        }
        for field_name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        if self.benchmark_warmup_requests < 0:
            raise ValueError("benchmark_warmup_requests cannot be negative")


PROFILES: dict[ProfileName, Profile] = {
    ProfileName.DEV_SMOKE: Profile(
        name=ProfileName.DEV_SMOKE,
        calibration_samples=16,
        calibration_sequence_length=512,
        calibration_batch_size=1,
        mmlu_samples=30,
        gsm8k_samples=20,
        benchmark_requests=32,
        benchmark_warmup_requests=4,
        benchmark_repetitions=1,
        minimum_free_disk_gib=145,
    ),
    ProfileName.WORKSHOP_B200: Profile(
        name=ProfileName.WORKSHOP_B200,
        calibration_samples=128,
        calibration_sequence_length=512,
        calibration_batch_size=1,
        mmlu_samples=300,
        gsm8k_samples=100,
        benchmark_requests=256,
        benchmark_warmup_requests=16,
        benchmark_repetitions=3,
        minimum_free_disk_gib=145,
    ),
    ProfileName.FULL: Profile(
        name=ProfileName.FULL,
        calibration_samples=128,
        calibration_sequence_length=512,
        calibration_batch_size=1,
        mmlu_samples=1_000,
        gsm8k_samples=250,
        benchmark_requests=1_024,
        benchmark_warmup_requests=64,
        benchmark_repetitions=5,
        minimum_free_disk_gib=145,
    ),
}


def resolve_profile(value: ProfileName | str) -> Profile:
    """Resolve a profile without silently substituting an unknown name."""

    try:
        name = value if isinstance(value, ProfileName) else ProfileName(value.upper())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(item.value for item in ProfileName)
        raise ValueError(f"Unknown profile {value!r}; choose one of: {choices}") from exc
    return PROFILES[name]


def get_profile(value: ProfileName | str) -> Profile:
    """Public profile lookup used by preparation scripts and notebooks."""

    return resolve_profile(value)


@dataclass(frozen=True)
class WorkshopConfig:
    """Complete, serializable configuration that determines a run identity."""

    profile: Profile
    artifact_root: Path
    project_root: Path
    seed: int = SEED
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    modelopt_version: str = MODELOPT_VERSION
    modelopt_source_tag: str = MODELOPT_SOURCE_TAG
    modelopt_git_commit: str = MODELOPT_GIT_COMMIT
    calibration_dataset: str = CALIBRATION_DATASET
    source_dtype: str = "bf16"
    kv_cache_dtype: str = "bf16"

    def __post_init__(self) -> None:
        if self.model_id != MODEL_ID or self.model_revision != MODEL_REVISION:
            raise ValueError("The workshop model and revision are pinned and cannot be overridden")
        if self.modelopt_version != MODELOPT_VERSION:
            raise ValueError(f"ModelOpt must remain pinned to {MODELOPT_VERSION}")
        if self.modelopt_source_tag != MODELOPT_SOURCE_TAG:
            raise ValueError(f"ModelOpt source tag must remain pinned to {MODELOPT_SOURCE_TAG}")
        if self.modelopt_git_commit != MODELOPT_GIT_COMMIT:
            raise ValueError("The ModelOpt Git checkout must match the pinned release commit")
        if self.source_dtype != "bf16" or self.kv_cache_dtype != "bf16":
            raise ValueError("The controlled comparison requires BF16 source weights and BF16 KV cache")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")

    @property
    def recipe_dir(self) -> Path:
        return self.project_root / "configs" / "recipes"

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return only experiment-affecting values, excluding filesystem location."""

        profile_data = asdict(self.profile)
        profile_data["name"] = self.profile.name.value
        return {
            "schema_version": 1,
            "profile": profile_data,
            "seed": self.seed,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "modelopt_version": self.modelopt_version,
            "modelopt_source_tag": self.modelopt_source_tag,
            "modelopt_git_commit": self.modelopt_git_commit,
            "modelopt_nemotron_export_fix_commit": MODELOPT_NEMOTRON_EXPORT_FIX_COMMIT,
            "modelopt_example": MODELOPT_EXAMPLE.as_posix(),
            "tensorrt_llm_version": TRTLLM_VERSION,
            "container_image": TRTLLM_IMAGE,
            "calibration_dataset": self.calibration_dataset,
            "source_dtype": self.source_dtype,
            "kv_cache_dtype": self.kv_cache_dtype,
        }


def make_config(
    profile: ProfileName | str = ProfileName.WORKSHOP_B200,
    *,
    project_root: Path | str | None = None,
    artifact_root: Path | str | None = None,
) -> WorkshopConfig:
    """Create a run configuration with deterministic, explicit paths."""

    resolved_project_root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    resolved_artifact_root = (
        Path(artifact_root).expanduser().resolve()
        if artifact_root is not None
        else resolved_project_root / "artifacts" / "runs"
    )
    return WorkshopConfig(
        profile=resolve_profile(profile),
        artifact_root=resolved_artifact_root,
        project_root=resolved_project_root,
    )
