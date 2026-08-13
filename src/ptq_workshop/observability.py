"""Durable timing and resource observations for ModelOpt PTQ runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PROGRESS_ELAPSED = re.compile(
    r"\b(?P<completed>\d+)/(?P<total>\d+)\s*\["
    r"(?P<elapsed>\d+:[0-5]\d(?::[0-5]\d)?)<"
)
_EXPORT_DURATION = re.compile(
    r"Quantized model exported to:.*?Total time used\s+"
    r"(?P<seconds>\d+(?:\.\d+)?)s"
)


def _elapsed_seconds(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ValueError(f"Unsupported elapsed time: {value!r}")


def parse_modelopt_phase_durations(log: str | Path) -> dict[str, Any]:
    """Extract phase durations emitted by the pinned ModelOpt example.

    Calibration uses tqdm's final elapsed counter, which is integral-second
    resolution. Unified-HF export prints its own high-resolution timer. The
    returned source labels make that distinction explicit rather than implying
    both measurements have the precision of the outer wall clock.
    """

    path = Path(log)
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    clean = _ANSI_ESCAPE.sub("", text)
    quant_summary_position = clean.rfind("Quant summary saved")
    calibration_region = (
        clean[:quant_summary_position] if quant_summary_position >= 0 else clean
    )
    completed_progress = [
        match
        for match in _PROGRESS_ELAPSED.finditer(calibration_region)
        if match.group("completed") == match.group("total")
    ]
    export_matches = list(_EXPORT_DURATION.finditer(clean))
    calibration_duration = (
        _elapsed_seconds(completed_progress[-1].group("elapsed"))
        if completed_progress
        else None
    )
    export_duration = (
        float(export_matches[-1].group("seconds")) if export_matches else None
    )
    return {
        "calibration_duration_s": calibration_duration,
        "calibration_duration_source": (
            "modelopt_tqdm_final_elapsed" if calibration_duration is not None else None
        ),
        "export_duration_s": export_duration,
        "export_duration_source": (
            "modelopt_unified_hf_timer" if export_duration is not None else None
        ),
        "all_requested_phase_durations_available": (
            calibration_duration is not None and export_duration is not None
        ),
    }
