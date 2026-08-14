from __future__ import annotations

import json
from pathlib import Path

import pytest

from ptq_workshop.notebook_artifacts import (
    NotebookExecutionError,
    archive_executed_notebook,
    notebook_execution_summary,
)


def _notebook(*, executed: bool, error: bool = False) -> dict:
    outputs = (
        [{"output_type": "error", "ename": "RuntimeError", "evalue": "boom"}]
        if error
        else [{"output_type": "stream", "name": "stdout", "text": "ok\n"}]
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# title"],
            },
            {
                "cell_type": "code",
                "execution_count": 1 if executed else None,
                "metadata": {},
                "outputs": outputs if executed else [],
                "source": ["print('ok')"],
            },
        ],
    }


def test_archives_complete_notebook_immutably_and_idempotently(tmp_path: Path) -> None:
    notebook = tmp_path / "workshop.ipynb"
    notebook.write_text(json.dumps(_notebook(executed=True)), encoding="utf-8")
    first = archive_executed_notebook(notebook, tmp_path / "run")
    second = archive_executed_notebook(notebook, tmp_path / "run")
    assert first == second
    assert Path(first["archived_path"]).read_bytes() == notebook.read_bytes()
    assert first["execution"]["status"] == "complete"
    assert first["reran_cells"] is False


def test_partial_or_failed_notebook_is_preserved_before_failure(tmp_path: Path) -> None:
    notebook = tmp_path / "workshop.ipynb"
    notebook.write_text(json.dumps(_notebook(executed=True, error=True)), encoding="utf-8")
    with pytest.raises(NotebookExecutionError, match="status is failed"):
        archive_executed_notebook(notebook, tmp_path / "run")
    manifests = list((tmp_path / "run" / "notebooks").glob("*.json"))
    archives = list((tmp_path / "run" / "notebooks").glob("*.ipynb"))
    assert len(manifests) == len(archives) == 1
    assert json.loads(manifests[0].read_text())["execution"]["error_output_count"] == 1


def test_empty_source_code_cells_do_not_reduce_execution_coverage() -> None:
    notebook = _notebook(executed=True)
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [],
        }
    )
    assert notebook_execution_summary(notebook)["status"] == "complete"
