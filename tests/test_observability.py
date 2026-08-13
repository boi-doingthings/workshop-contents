from pathlib import Path

from ptq_workshop.observability import parse_modelopt_phase_durations


def test_parses_pinned_modelopt_calibration_and_export_durations(tmp_path: Path) -> None:
    log = tmp_path / "quantize.log"
    log.write_text(
        "Loading weights: 401/401 [00:02<00:00, 140it/s]\r"
        "16/16 [01:07<00:00, 4.1s/it]\r"
        "\x1b[1mQuant summary saved to /export/.quant_summary.txt\x1b[0m\n"
        "Quantized model exported to: /export. Total time used 13.125s\n",
        encoding="utf-8",
    )
    result = parse_modelopt_phase_durations(log)
    assert result == {
        "calibration_duration_s": 67.0,
        "calibration_duration_source": "modelopt_tqdm_final_elapsed",
        "export_duration_s": 13.125,
        "export_duration_source": "modelopt_unified_hf_timer",
        "all_requested_phase_durations_available": True,
    }


def test_missing_phase_markers_remain_explicitly_unavailable(tmp_path: Path) -> None:
    result = parse_modelopt_phase_durations(tmp_path / "missing.log")
    assert result["calibration_duration_s"] is None
    assert result["export_duration_s"] is None
    assert result["all_requested_phase_durations_available"] is False
