import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

from ptq_workshop.artifacts import ArtifactLayout
from ptq_workshop.config import MODELOPT_EXAMPLE, make_config
from ptq_workshop.quantize import (
    CLEAN_QUANTIZATION_EXCLUDE_MODULES,
    CONFIG_NORMALIZATION_MANIFEST,
    CONTROLLED_EXPORT_BF16_WEIGHTS,
    CONTROLLED_IN_MEMORY_QUANTIZERS,
    CONTROLLED_RECIPE_EXCLUSION_PATTERNS,
    MAMBA_LAYER_INDICES,
    PINNED_HYBRID_OVERRIDE_PATTERN,
    PINNED_NUM_HIDDEN_LAYERS,
    POSITIVE_CONTROL_EXPORT_MODULE,
    POSITIVE_CONTROL_IN_MEMORY_QUANTIZERS,
    POSITIVE_CONTROL_SOURCE_SHAPE,
    POSITIVE_CONTROL_SOURCE_WEIGHT,
    QuantizationError,
    QUANTIZATION_METADATA_NORMALIZATION_MANIFEST,
    SOURCE_TOPOLOGY_NORMALIZATION_MANIFEST,
    assert_controlled_recipes,
    assert_controlled_topology,
    build_quantization_job,
    disabled_quantizers,
    load_recipe,
    normalize_export_config,
    normalize_quantization_metadata,
    normalize_source_topology,
    validate_calibration_jsonl,
    validate_export,
    validate_quant_summary,
)


def test_normalize_export_config_removes_only_read_only_derived_field(
    tmp_path: Path,
) -> None:
    export = tmp_path / "fp8"
    export.mkdir()
    config_path = export / "config.json"
    original = {
        "model_type": "nemotron_h",
        "layers_block_type": ["mamba", "attention"],
        "mtp_layers_block_type": ["attention", "moe"],
        "dtype": "bfloat16",
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"},
    }
    config_path.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")

    audit = normalize_export_config(export)

    normalized = json.loads(config_path.read_text(encoding="utf-8"))
    assert "layers_block_type" not in normalized
    assert normalized["mtp_layers_block_type"] == original["mtp_layers_block_type"]
    assert normalized["quantization_config"] == original["quantization_config"]
    assert audit["before_sha256"] != audit["after_sha256"]
    assert audit["removed_fields"]["layers_block_type"]["value"] == [
        "mamba",
        "attention",
    ]
    manifest = json.loads(
        (export / CONFIG_NORMALIZATION_MANIFEST).read_text(encoding="utf-8")
    )
    assert manifest["after_sha256"] == audit["after_sha256"]
    assert manifest["quantization_config_sha256"] == audit["quantization_config_sha256"]

    audit_path = export / CONFIG_NORMALIZATION_MANIFEST
    audit_bytes = audit_path.read_bytes()
    repeated = normalize_export_config(export)
    assert audit_path.read_bytes() == audit_bytes
    assert repeated == audit


def test_normalize_export_config_rejects_normalized_config_without_original_audit(
    tmp_path: Path,
) -> None:
    export = tmp_path / "orphan"
    export.mkdir()
    (export / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_h",
                "quantization_config": {"quant_method": "modelopt"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(QuantizationError, match="lacks original removal evidence"):
        normalize_export_config(export)


def test_normalize_export_config_rejects_weakened_existing_audit(tmp_path: Path) -> None:
    export = tmp_path / "weakened"
    export.mkdir()
    config_path = export / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "layers_block_type": ["mamba"],
                "quantization_config": {"quant_method": "modelopt"},
            }
        ),
        encoding="utf-8",
    )
    normalize_export_config(export)
    audit_path = export / CONFIG_NORMALIZATION_MANIFEST
    audit = json.loads(audit_path.read_text())
    audit["removed_fields"] = {}
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(QuantizationError, match="prove removal of exactly"):
        normalize_export_config(export)


def test_export_validator_rejects_unsettable_layers_block_type(tmp_path: Path) -> None:
    export = tmp_path / "unsafe"
    export.mkdir()
    (export / "config.json").write_text(
        json.dumps(
            {
                "layers_block_type": ["mamba"],
                "quantization_config": {"quant_method": "modelopt"},
            }
        ),
        encoding="utf-8",
    )
    (export / "hf_quant_config.json").write_text("{}\n", encoding="utf-8")
    save_file({"weight": torch.ones(1)}, export / "model.safetensors")

    with pytest.raises(QuantizationError, match="read-only Nemotron-H"):
        validate_export(export)


def test_export_validator_requires_current_config_normalization_audit(
    tmp_path: Path,
) -> None:
    export = tmp_path / "missing-audit"
    export.mkdir()
    model_config = _export_model_config()
    model_config.pop("layers_block_type")
    (export / "config.json").write_text(json.dumps(model_config) + "\n", encoding="utf-8")
    (export / "hf_quant_config.json").write_text("{}\n", encoding="utf-8")
    save_file({"weight": torch.ones(1)}, export / "model.safetensors")
    with pytest.raises(QuantizationError, match="lacks config-normalization audit"):
        validate_export(export)


HYBRID_PATTERN = PINNED_HYBRID_OVERRIDE_PATTERN


def _model_config() -> dict[str, object]:
    return {
        "hybrid_override_pattern": HYBRID_PATTERN,
        "num_hidden_layers": len(HYBRID_PATTERN),
    }


def _export_model_config() -> dict[str, object]:
    return {
        "layers_block_type": ["mamba"] * PINNED_NUM_HIDDEN_LAYERS,
        "quantization_config": {
            "quant_method": "modelopt",
            "ignore": ["lm_head.\x00backbone.pt_name_sentinel"],
        },
    }


def test_recipes_have_identical_exclusions_and_no_kv_quantizers() -> None:
    root = Path(__file__).resolve().parents[1]
    fp8 = root / "configs" / "recipes" / "fp8.yaml"
    nvfp4 = root / "configs" / "recipes" / "nvfp4.yaml"
    assert_controlled_recipes(fp8, nvfp4)
    assert disabled_quantizers(load_recipe(fp8)) == disabled_quantizers(load_recipe(nvfp4))
    for recipe in (fp8, nvfp4):
        text = recipe.read_text(encoding="utf-8")
        assert "bmm_quantizer" not in text
        assert "*mixer.conv1d*" in text
        assert "'*backbone.layers" not in text
        assert "*model.layers.4.mixer.in_proj*" in text
        assert "*model.layers.42.mixer.o_proj*" in text
        assert all(pattern in text for pattern in CONTROLLED_RECIPE_EXCLUSION_PATTERNS)


def test_controlled_topology_is_derived_from_the_model_config() -> None:
    assert_controlled_topology(_model_config(), location="fixture")
    drifted = _model_config()
    drifted["hybrid_override_pattern"] = HYBRID_PATTERN.replace("*", "M", 1)
    with pytest.raises(QuantizationError, match="differs from the pinned"):
        assert_controlled_topology(drifted, location="fixture")


def test_recipe_control_rejects_different_exclusions(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "recipes"
    fp8 = tmp_path / "fp8.yaml"
    nvfp4 = tmp_path / "nvfp4.yaml"
    fp8.write_text((root / "fp8.yaml").read_text(), encoding="utf-8")
    recipe = yaml.safe_load((root / "nvfp4.yaml").read_text())
    recipe["quantize"]["quant_cfg"].append({"quantizer_name": "*extra*", "enable": False})
    nvfp4.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="identical"):
        assert_controlled_recipes(fp8, nvfp4)


def test_recipe_control_rejects_identically_missing_required_exclusion(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "recipes"
    paths = [tmp_path / "fp8.yaml", tmp_path / "nvfp4.yaml"]
    missing_pattern = CONTROLLED_RECIPE_EXCLUSION_PATTERNS[0]
    for source_name, destination in zip(("fp8.yaml", "nvfp4.yaml"), paths):
        recipe = yaml.safe_load((root / source_name).read_text())
        recipe["quantize"]["quant_cfg"] = [
            entry
            for entry in recipe["quantize"]["quant_cfg"]
            if entry.get("quantizer_name") != missing_pattern
        ]
        destination.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="exact in-memory Nemotron"):
        assert_controlled_recipes(*paths)


def test_command_uses_same_local_bf16_source_and_frozen_jsonl(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = make_config("DEV_SMOKE", project_root=project_root, artifact_root=tmp_path / "runs")
    layout = ArtifactLayout(config.artifact_root, "0" * 16)
    source = tmp_path / "hub" / "snapshots" / config.model_revision
    source.mkdir(parents=True)
    (source / "config.json").write_text(json.dumps(_model_config()), encoding="utf-8")
    (source / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        "".join(json.dumps({"text": f"sample {index}"}) + "\n" for index in range(16)),
        encoding="utf-8",
    )
    modelopt = tmp_path / "Model-Optimizer"
    script = modelopt / MODELOPT_EXAMPLE
    script.parent.mkdir(parents=True)
    script.write_text("# fixture\n", encoding="utf-8")
    jobs = [
        build_quantization_job(
            config,
            layout,
            variant=variant,
            source_model=source,
            calibration_jsonl=calibration,
            modelopt_root=modelopt,
        )
        for variant in ("fp8", "nvfp4")
    ]
    assert jobs[0].source_model == jobs[1].source_model == source.resolve()
    for job in jobs:
        assert job.command[1].endswith("examples/hf_ptq/hf_ptq.py")
        assert "--recipe" in job.command
        assert "--dataset" in job.command
        assert str(calibration.resolve()) in job.command
        assert "--kv_cache_qformat" not in job.command
        assert "--skip_generate" in job.command


def test_frozen_calibration_count_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "calibration.jsonl"
    path.write_text('{"text":"one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 2"):
        validate_calibration_jsonl(path, 2)


def _write_quant_summary(
    path: Path,
    *,
    enabled_name: str | None = None,
    variant: str = "fp8",
) -> None:
    positive_reprs = {
        "fp8": (
            "TensorQuantizer((4, 3) bit fake per-tensor amax=1.00e+00 "
            "calibrator=MaxCalibrator quant)"
        ),
        "nvfp4": (
            "TensorQuantizer((2, 1) bit fake block_sizes={-1: 16, 'type': 'dynamic', "
            "'scale_bits': (4, 3)}, amax=1.00e+00 calibrator=MaxCalibrator quant)"
        ),
    }
    path.write_text(
        "".join(
            f"{name:<84} TensorQuantizer("
            f"{'quant' if name == enabled_name else 'disabled'})\n"
            for name in CONTROLLED_IN_MEMORY_QUANTIZERS
        )
        + "".join(
            f"{name:<84} {positive_reprs[variant]}\n"
            for name in POSITIVE_CONTROL_IN_MEMORY_QUANTIZERS
        ),
        encoding="utf-8",
    )


def test_quant_summary_proves_exact_live_quantizers_are_disabled(tmp_path: Path) -> None:
    summary = tmp_path / ".quant_summary.txt"
    _write_quant_summary(summary)
    proof = validate_quant_summary(summary, expected_variant="fp8")
    assert proof["controlled_disabled_quantizer_count"] == 72
    assert proof["positive_control_quantizer_count"] == 2
    assert proof["positive_control_variant"] == "fp8"

    _write_quant_summary(summary, enabled_name=CONTROLLED_IN_MEMORY_QUANTIZERS[0])
    with pytest.raises(QuantizationError, match="not disabled"):
        validate_quant_summary(summary)

    _write_quant_summary(summary, variant="nvfp4")
    with pytest.raises(QuantizationError, match="format mismatch"):
        validate_quant_summary(summary, expected_variant="fp8")


def _write_export(
    path: Path,
    *,
    variant: str = "fp8",
    bad_weight: str | None = None,
    bad_shape: str | None = None,
    unexpected_scale: str | None = None,
    normalize_topology: bool = True,
) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(_export_model_config()) + "\n", encoding="utf-8"
    )
    normalize_export_config(path)
    (path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt", "version": "fixture"},
                "quantization": {
                    "quant_algo": "FP8" if variant == "fp8" else "NVFP4",
                    "exclude_modules": ["lm_head.\x00backbone.pt_name_sentinel"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_quant_summary(path / ".quant_summary.txt", variant=variant)
    tensors = {
        name: torch.ones(
            2 if name == bad_shape else 1,
            dtype=torch.float16 if name == bad_weight else torch.bfloat16,
        )
        for name in CONTROLLED_EXPORT_BF16_WEIGHTS
    }
    if variant == "fp8":
        tensors[f"{POSITIVE_CONTROL_EXPORT_MODULE}.weight"] = torch.zeros(
            POSITIVE_CONTROL_SOURCE_SHAPE, dtype=torch.float8_e4m3fn
        )
        tensors[f"{POSITIVE_CONTROL_EXPORT_MODULE}.input_scale"] = torch.ones(
            (), dtype=torch.float32
        )
        tensors[f"{POSITIVE_CONTROL_EXPORT_MODULE}.weight_scale"] = torch.ones(
            (), dtype=torch.float32
        )
    else:
        tensors[f"{POSITIVE_CONTROL_EXPORT_MODULE}.weight"] = torch.zeros(
            (POSITIVE_CONTROL_SOURCE_SHAPE[0], POSITIVE_CONTROL_SOURCE_SHAPE[1] // 2),
            dtype=torch.uint8,
        )
        tensors[f"{POSITIVE_CONTROL_EXPORT_MODULE}.input_scale"] = torch.ones(
            (), dtype=torch.float32
        )
        tensors[f"{POSITIVE_CONTROL_EXPORT_MODULE}.weight_scale"] = torch.ones(
            (POSITIVE_CONTROL_SOURCE_SHAPE[0], POSITIVE_CONTROL_SOURCE_SHAPE[1] // 16),
            dtype=torch.float8_e4m3fn,
        )
        tensors[f"{POSITIVE_CONTROL_EXPORT_MODULE}.weight_scale_2"] = torch.ones(
            (), dtype=torch.float32
        )
    if unexpected_scale is not None:
        tensors[unexpected_scale] = torch.ones(1, dtype=torch.float32)
    tensors["lm_head.weight"] = torch.ones(1, dtype=torch.bfloat16)
    for layer in MAMBA_LAYER_INDICES:
        tensors[f"backbone.layers.{layer}.mixer.conv1d.weight"] = torch.ones(
            1, dtype=torch.bfloat16
        )
    save_file(tensors, path / "model.safetensors")
    normalize_quantization_metadata(path)
    if normalize_topology:
        source_fixture = path / ".source_fixture"
        source_fixture.mkdir()
        (source_fixture / "config.json").write_text(
            json.dumps(_model_config()) + "\n", encoding="utf-8"
        )
        normalize_source_topology(path, source_model=source_fixture)


def _write_source(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_model_config()) + "\n", encoding="utf-8")
    tensors = {
        name: torch.ones(1, dtype=torch.bfloat16)
        for name in CONTROLLED_EXPORT_BF16_WEIGHTS
    }
    tensors[POSITIVE_CONTROL_SOURCE_WEIGHT] = torch.zeros(
        POSITIVE_CONTROL_SOURCE_SHAPE, dtype=torch.bfloat16
    )
    save_file(tensors, path / "model.safetensors")


def test_quantization_metadata_normalization_is_exact_and_idempotent(
    tmp_path: Path,
) -> None:
    export = tmp_path / "fp8-metadata"
    _write_export(export)
    audit_path = export / QUANTIZATION_METADATA_NORMALIZATION_MANIFEST
    before = audit_path.read_bytes()

    audit = normalize_quantization_metadata(export)

    assert audit_path.read_bytes() == before
    config = json.loads((export / "config.json").read_text())
    hf_config = json.loads((export / "hf_quant_config.json").read_text())
    expected = list(CLEAN_QUANTIZATION_EXCLUDE_MODULES)
    assert len(expected) == 60
    assert config["quantization_config"]["ignore"] == expected
    assert hf_config["quantization"]["exclude_modules"] == expected
    assert hf_config["quantization"]["torch_dtype"] == "bfloat16"
    assert not any("\x00" in module for module in expected)
    assert audit["tensor_backed_exclusion_count"] == 60


def test_quantization_metadata_normalization_requires_tensor_backing(
    tmp_path: Path,
) -> None:
    export = tmp_path / "missing-tensor"
    export.mkdir()
    (export / "config.json").write_text(
        json.dumps(_export_model_config()) + "\n", encoding="utf-8"
    )
    normalize_export_config(export)
    (export / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt"},
                "quantization": {
                    "quant_algo": "FP8",
                    "exclude_modules": ["lm_head.\x00backbone.pt_name_sentinel"],
                },
            }
        ),
        encoding="utf-8",
    )
    save_file({"lm_head.weight": torch.ones(1)}, export / "model.safetensors")
    with pytest.raises(QuantizationError, match="not backed by serialized tensors"):
        normalize_quantization_metadata(export)


def test_source_topology_normalization_restores_only_pinned_fields_and_is_idempotent(
    tmp_path: Path,
) -> None:
    export = tmp_path / "topology"
    _write_export(export)
    audit_path = export / SOURCE_TOPOLOGY_NORMALIZATION_MANIFEST
    audit_bytes = audit_path.read_bytes()
    config = json.loads((export / "config.json").read_text())
    audit = json.loads(audit_bytes)

    repeated = normalize_source_topology(
        export, source_model=export / ".source_fixture"
    )

    assert audit_path.read_bytes() == audit_bytes
    assert repeated["config_before_sha256"] == audit["config_before_sha256"]
    assert config["hybrid_override_pattern"] == PINNED_HYBRID_OVERRIDE_PATTERN
    assert config["num_hidden_layers"] == PINNED_NUM_HIDDEN_LAYERS
    assert set(audit["restored_fields"]) == {
        "hybrid_override_pattern",
        "num_hidden_layers",
    }
    assert audit["tensor_metadata_before_sha256"] == audit[
        "tensor_metadata_after_sha256"
    ]
    # Earlier normalization stages must also remain idempotent through the
    # complete config -> quant metadata -> topology audit chain.
    normalize_export_config(export)
    normalize_quantization_metadata(export)


def test_source_topology_normalization_rejects_drifted_source(tmp_path: Path) -> None:
    export = tmp_path / "drifted-topology"
    _write_export(export, normalize_topology=False)
    source = export / "source"
    source.mkdir()
    drifted = _model_config()
    drifted["hybrid_override_pattern"] = HYBRID_PATTERN.replace("E", "M", 1)
    (source / "config.json").write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(QuantizationError, match="differs from the pinned"):
        normalize_source_topology(export, source_model=source)


def test_export_validator_checks_physical_bf16_tensors_not_only_config(tmp_path: Path) -> None:
    export = tmp_path / "valid"
    _write_export(export)
    proof = validate_export(export, expected_variant="fp8")
    assert proof["controlled_bf16_weight_count"] == 36
    assert set(proof["controlled_bf16_weight_dtypes"].values()) == {"torch.bfloat16"}
    assert proof["positive_control_variant"] == "fp8"
    assert proof["positive_control_tensor_count"] == 3

    invalid = tmp_path / "invalid"
    bad_weight = CONTROLLED_EXPORT_BF16_WEIGHTS[0]
    _write_export(invalid, bad_weight=bad_weight)
    with pytest.raises(QuantizationError, match="expected torch.bfloat16"):
        validate_export(invalid, expected_variant="fp8")

    scaled = tmp_path / "scaled"
    controlled_module = CONTROLLED_EXPORT_BF16_WEIGHTS[0].removesuffix(".weight")
    _write_export(scaled, unexpected_scale=f"{controlled_module}.unlisted_calibration_amax")
    with pytest.raises(QuantizationError, match="unexpectedly has quantization scale/amax tensor"):
        validate_export(scaled, expected_variant="fp8")


def test_export_validator_proves_nvfp4_positive_control_metadata(tmp_path: Path) -> None:
    export = tmp_path / "nvfp4"
    _write_export(export, variant="nvfp4")
    proof = validate_export(export, expected_variant="nvfp4")
    assert proof["positive_control_variant"] == "nvfp4"
    assert proof["positive_control_tensor_count"] == 4


def test_export_validator_compares_controlled_shapes_to_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)

    valid = tmp_path / "valid-with-source"
    _write_export(valid)
    proof = validate_export(valid, expected_variant="fp8", source_model=source)
    assert len(proof["controlled_bf16_weight_shapes"]) == 36

    invalid = tmp_path / "bad-shape"
    bad_shape = CONTROLLED_EXPORT_BF16_WEIGHTS[0]
    _write_export(invalid, bad_shape=bad_shape)
    with pytest.raises(QuantizationError, match="shapes differ from the pinned source"):
        validate_export(invalid, expected_variant="fp8", source_model=source)
