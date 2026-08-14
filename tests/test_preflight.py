from pathlib import Path
from shutil import _ntuple_diskusage

from ptq_workshop.config import get_profile
from ptq_workshop.preflight import (
    GIB,
    disk_space_check,
    is_supported_compute_capability,
    parse_nvidia_smi_csv,
)


def test_blackwell_compute_capability_accepts_b200_b300_and_rtx_blackwell() -> None:
    assert is_supported_compute_capability("10.0")
    assert is_supported_compute_capability("10.3")
    assert is_supported_compute_capability((12, 0))
    assert not is_supported_compute_capability((12, 1))
    assert not is_supported_compute_capability("8.9")


def test_full_gpu_audit_row_parses() -> None:
    row = (
        "0, NVIDIA B200, 10.0, 183456, GPU-abc, 590.44, 12, 1, 31, "
        "82, 1000, 1980, 6000, Enabled, Not Active, Not Active\n"
    )
    gpu = parse_nvidia_smi_csv(row)[0]
    assert gpu.uuid == "GPU-abc"
    assert gpu.memory_total_mib == 183456
    assert gpu.power_limit_w == 1000
    assert gpu.ecc_mode == "Enabled"


def test_b300_gpu_audit_row_is_supported() -> None:
    gpu = parse_nvidia_smi_csv("0, NVIDIA B300, 10.3, 294912\n")[0]
    assert gpu.compute_capability == (10, 3)
    assert is_supported_compute_capability(gpu.compute_capability)


def test_disk_check_is_inclusive_and_145_is_only_an_uncached_capacity_hint(tmp_path: Path) -> None:
    usage = lambda _: _ntuple_diskusage(200 * GIB, 55 * GIB, 145 * GIB)
    assert disk_space_check(tmp_path, 145, usage=usage).passed
    assert not disk_space_check(tmp_path, 146, usage=usage).passed
    for name in ("DEV_SMOKE", "WORKSHOP_B200", "FULL"):
        assert get_profile(name).initial_uncached_capacity_gib == 145
