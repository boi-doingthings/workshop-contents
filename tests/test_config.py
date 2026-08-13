from pathlib import Path

import pytest

from ptq_workshop.config import (
    MODEL_ID,
    MODEL_REVISION,
    MODELOPT_EXAMPLE,
    MODELOPT_GIT_COMMIT,
    MODELOPT_NEMOTRON_EXPORT_FIX_COMMIT,
    MODELOPT_SOURCE_TAG,
    MODELOPT_VERSION,
    ProfileName,
    WorkshopConfig,
    get_profile,
    make_config,
)


def test_profiles_match_workshop_contract() -> None:
    dev = get_profile("dev_smoke")
    workshop = get_profile(ProfileName.WORKSHOP_B200)
    full = get_profile("FULL")
    assert (dev.calibration_samples, dev.calibration_sequence_length) == (16, 512)
    assert (workshop.calibration_samples, workshop.mmlu_samples, workshop.gsm8k_samples) == (
        128,
        300,
        100,
    )
    assert (full.calibration_samples, full.mmlu_samples, full.gsm8k_samples) == (128, 1000, 250)
    assert all(profile.minimum_free_disk_gib == 145 for profile in (dev, workshop, full))
    assert workshop.model_id == MODEL_ID
    assert workshop.model_revision == MODEL_REVISION
    assert workshop.seed == 42


def test_unknown_profile_fails_without_fallback() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        get_profile("fastish")


def test_config_rejects_model_override(tmp_path: Path) -> None:
    config = make_config("DEV_SMOKE", project_root=tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        WorkshopConfig(
            profile=config.profile,
            artifact_root=config.artifact_root,
            project_root=config.project_root,
            model_id="another/model",
        )


def test_default_artifacts_live_under_runs(tmp_path: Path) -> None:
    config = make_config("DEV_SMOKE", project_root=tmp_path)
    assert config.artifact_root == tmp_path / "artifacts" / "runs"


def test_modelopt_pin_contains_nemotron_export_fix() -> None:
    config = make_config("DEV_SMOKE")
    payload = config.fingerprint_payload()
    assert MODELOPT_VERSION == MODELOPT_SOURCE_TAG == "0.46.0rc0"
    assert MODELOPT_GIT_COMMIT == "33d05b0c446f528914173041057050f6d135fbf4"
    assert MODELOPT_NEMOTRON_EXPORT_FIX_COMMIT == "c81210faecc096a7bd802cca2cda909ac43f7759"
    assert MODELOPT_EXAMPLE.as_posix() == "examples/hf_ptq/hf_ptq.py"
    assert payload["modelopt_source_tag"] == MODELOPT_SOURCE_TAG
    assert payload["modelopt_nemotron_export_fix_commit"] == MODELOPT_NEMOTRON_EXPORT_FIX_COMMIT


def test_bootstrap_installs_verified_modelopt_source_non_editably() -> None:
    root = Path(__file__).resolve().parents[1]
    bootstrap = (root / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert 'MODELOPT_TAG="0.46.0rc0"' in bootstrap
    assert f'MODELOPT_COMMIT="{MODELOPT_GIT_COMMIT}"' in bootstrap
    assert 'git -C "${MODELOPT_DIR}" rev-parse HEAD' in bootstrap
    assert 'pip install --no-deps --force-reinstall "${MODELOPT_DIR}"' in bootstrap
    assert 'pip install --no-deps -e "${MODELOPT_DIR}"' not in bootstrap
    assert "Mixed ModelOpt namespace" in bootstrap


def test_bootstrap_uses_shared_b200_b300_hardware_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    bootstrap = (root / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert "from ptq_workshop.preflight import is_supported_compute_capability" in bootstrap
    assert "assert is_supported_compute_capability(capability)" in bootstrap
    assert "torch.cuda.get_device_capability(0) in {(10, 0), (12, 0)}" not in bootstrap
