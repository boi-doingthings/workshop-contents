from pathlib import Path
from shutil import _ntuple_diskusage

from ptq_workshop.artifacts import ArtifactLayout
from ptq_workshop.storage import GIB, plan_run_storage


def _usage(free_gib: int):
    return lambda _: _ntuple_diskusage(200 * GIB, (200 - free_gib) * GIB, free_gib * GIB)


def test_resume_credits_existing_packed_checkpoints(tmp_path: Path) -> None:
    layout = ArtifactLayout(tmp_path / "runs", "a" * 16)
    for variant, gib in (("fp8", 31), ("nvfp4", 18)):
        directory = layout.checkpoint_dir(variant)
        directory.mkdir(parents=True)
        # Mock allocated size because sparse test files should not consume GiBs.
        (directory / "model.safetensors").write_bytes(b"packed")
        (directory / "config.json").write_text("{}")
        (directory / "hf_quant_config.json").write_text("{}")
        (directory / "model.safetensors.index.json").write_text("{}")

    import ptq_workshop.storage as storage

    original = storage.allocated_bytes
    storage.allocated_bytes = lambda path: {
        layout.checkpoint_dir("fp8"): 31 * GIB,
        layout.checkpoint_dir("nvfp4"): 18 * GIB,
    }.get(Path(path), original(path))
    try:
        plan = plan_run_storage(
            layout, stage="preflight", precision="all", usage=_usage(15)
        )
    finally:
        storage.allocated_bytes = original
    assert plan.required_free_gib == 6
    assert plan.passed


def test_clean_overwrite_credits_only_selected_run(tmp_path: Path) -> None:
    layout = ArtifactLayout(tmp_path / "runs", "b" * 16)
    layout.run_dir.mkdir(parents=True)
    import ptq_workshop.storage as storage

    original = storage.allocated_bytes
    storage.allocated_bytes = lambda path: 49 * GIB if Path(path) == layout.run_dir else 0
    try:
        plan = plan_run_storage(
            layout,
            stage="all",
            precision="all",
            overwrite_run=True,
            usage=_usage(15),
        )
    finally:
        storage.allocated_bytes = original
    assert plan.projected_free_bytes == 64 * GIB
    assert plan.required_free_gib == 55
    assert plan.passed
