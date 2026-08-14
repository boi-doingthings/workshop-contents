"""Cache-aware, operation-scoped storage planning for repeatable demos."""

from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .config import (
    FRESH_EXPORTS_MINIMUM_FREE_DISK_GIB,
    RESUME_RESULTS_MINIMUM_FREE_DISK_GIB,
)


GIB = 1024**3
PARTIAL_EXPORT_BUDGET_GIB = {"fp8": 36, "nvfp4": 24}


def allocated_bytes(path: Path | str) -> int:
    """Return filesystem blocks reclaimable by deleting exactly ``path``."""

    target = Path(path)
    if not target.exists() or target.is_symlink():
        return 0
    total = target.stat().st_blocks * 512
    if target.is_dir():
        for item in target.rglob("*"):
            if not item.is_symlink():
                total += item.stat().st_blocks * 512
    return total


@dataclass(frozen=True)
class StoragePlan:
    stage: str
    precision: str
    overwrite_run: bool
    physical_free_bytes: int
    reclaimable_run_bytes: int
    reusable_checkpoint_bytes: int
    required_free_bytes: int

    @property
    def projected_free_bytes(self) -> int:
        return self.physical_free_bytes + (
            self.reclaimable_run_bytes if self.overwrite_run else 0
        )

    @property
    def passed(self) -> bool:
        return self.projected_free_bytes >= self.required_free_bytes

    @property
    def required_free_gib(self) -> int:
        return max(1, math.ceil(self.required_free_bytes / GIB))

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "projected_free_bytes": self.projected_free_bytes,
                "required_free_gib": self.required_free_gib,
                "passed": self.passed,
                "detail": (
                    f"{self.physical_free_bytes / GIB:.1f} GiB physically free; "
                    f"{self.reclaimable_run_bytes / GIB:.1f} GiB reclaimable from selected run; "
                    f"{self.reusable_checkpoint_bytes / GIB:.1f} GiB reusable checkpoints; "
                    f"{self.projected_free_bytes / GIB:.1f} GiB projected; "
                    f"{self.required_free_bytes / GIB:.1f} GiB required"
                ),
            }
        )
        return payload


def _selected_variants(precision: str) -> tuple[str, ...]:
    if precision == "all":
        return ("fp8", "nvfp4")
    if precision in PARTIAL_EXPORT_BUDGET_GIB:
        return (precision,)
    return ()


def _reusable_checkpoint_bytes(layout: Any, variant: str) -> int:
    checkpoint = Path(layout.checkpoint_dir(variant))
    required = (
        checkpoint / "config.json",
        checkpoint / "hf_quant_config.json",
        checkpoint / "model.safetensors.index.json",
    )
    if not all(path.is_file() for path in required) or not any(
        checkpoint.glob("*.safetensors")
    ):
        return 0
    return allocated_bytes(checkpoint)


def plan_run_storage(
    layout: Any,
    *,
    stage: str,
    precision: str,
    overwrite_run: bool = False,
    usage: Callable[[Path | str], shutil._ntuple_diskusage] = shutil.disk_usage,
) -> StoragePlan:
    """Budget only new bytes for the selected stage and exact run."""

    run_dir = Path(layout.run_dir)
    probe = run_dir.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    physical_free = usage(probe).free
    reclaimable = allocated_bytes(run_dir) if overwrite_run else 0
    variants = _selected_variants(precision)
    reusable = 0
    if not overwrite_run:
        reusable = sum(_reusable_checkpoint_bytes(layout, name) for name in variants)

    if stage in {"preflight", "quantize", "all"} and variants:
        fresh_gib = (
            FRESH_EXPORTS_MINIMUM_FREE_DISK_GIB
            if precision == "all"
            else PARTIAL_EXPORT_BUDGET_GIB[precision]
        )
        required = max(
            RESUME_RESULTS_MINIMUM_FREE_DISK_GIB * GIB,
            fresh_gib * GIB - reusable,
        )
    else:
        required = RESUME_RESULTS_MINIMUM_FREE_DISK_GIB * GIB
    return StoragePlan(
        stage=stage,
        precision=precision,
        overwrite_run=overwrite_run,
        physical_free_bytes=physical_free,
        reclaimable_run_bytes=reclaimable,
        reusable_checkpoint_bytes=reusable,
        required_free_bytes=required,
    )
