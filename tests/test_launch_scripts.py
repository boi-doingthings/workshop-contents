from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_notebook_launcher_exposes_explicit_instructor_controls() -> None:
    launcher = (ROOT / "scripts" / "launch_notebook.sh").read_text(encoding="utf-8")
    for option in ("--profile", "--live", "--dry-run", "--fresh", "--resume", "--port"):
        assert option in launcher
    assert "PTQ_PUBLISH_JUPYTER=1" in launcher
    assert "python -m jupyterlab" in launcher


def test_container_publishes_jupyter_on_loopback_only_and_forwards_controls() -> None:
    container = (ROOT / "scripts" / "container.sh").read_text(encoding="utf-8")
    assert '127.0.0.1:${JUPYTER_PORT_VALUE}:${JUPYTER_PORT_VALUE}' in container
    assert '"${PTQ_PUBLISH_JUPYTER:-0}" == "1"' in container
    for variable in ("PTQ_PROFILE", "PTQ_DRY_RUN", "PTQ_START_FRESH", "JUPYTER_PORT"):
        assert f"--env {variable}=" in container
