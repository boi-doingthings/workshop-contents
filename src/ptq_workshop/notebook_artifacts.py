"""Archive an already-executed notebook without rerunning any cells."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .artifacts import write_json_atomic


class NotebookExecutionError(RuntimeError):
    """Raised after an incomplete or failed notebook has still been archived."""


def notebook_execution_summary(notebook: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize saved execution coverage and error outputs."""

    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("notebook must contain a cells list")
    code_cells = [
        (index, cell)
        for index, cell in enumerate(cells)
        if isinstance(cell, Mapping)
        and cell.get("cell_type") == "code"
        and str("".join(cell.get("source", []))).strip()
    ]
    executed = [index for index, cell in code_cells if cell.get("execution_count") is not None]
    errors: list[dict[str, Any]] = []
    for index, cell in code_cells:
        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            raise ValueError(f"notebook cell {index} outputs must be a list")
        for output in outputs:
            if isinstance(output, Mapping) and output.get("output_type") == "error":
                errors.append(
                    {
                        "cell_index": index,
                        "ename": str(output.get("ename", "")),
                        "evalue": str(output.get("evalue", "")),
                    }
                )
    unexecuted = [index for index, _ in code_cells if index not in executed]
    if errors:
        status = "failed"
    elif unexecuted:
        status = "partial" if executed else "unexecuted"
    else:
        status = "complete"
    return {
        "status": status,
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": len(executed),
        "unexecuted_code_cell_indices": unexecuted,
        "error_output_count": len(errors),
        "error_outputs": errors,
    }


def _copy_bytes_immutable(destination: Path, content: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite different artifact: {destination}")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def archive_executed_notebook(
    notebook_path: str | Path,
    run_dir: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Hash-address and archive the saved notebook, including failed/partial runs.

    Validation happens before the optional exception but copying happens first,
    so a failed live session remains inspectable. The archive never executes a
    cell and never replaces a different notebook artifact.
    """

    source = Path(notebook_path).expanduser().resolve()
    destination_root = Path(run_dir).expanduser().resolve() / "notebooks"
    try:
        content = source.read_bytes()
        notebook = json.loads(content)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid notebook: {source}") from exc
    if not isinstance(notebook, Mapping) or int(notebook.get("nbformat", 0)) != 4:
        raise ValueError(f"Expected a Jupyter nbformat 4 notebook: {source}")
    summary = notebook_execution_summary(notebook)
    digest = hashlib.sha256(content).hexdigest()
    archived_path = destination_root / f"{source.stem}-executed-{digest[:16]}.ipynb"
    _copy_bytes_immutable(archived_path, content)
    manifest = {
        "schema_version": 1,
        "kind": "executed_notebook_archive",
        "source_path": str(source),
        "archived_path": str(archived_path),
        "sha256": digest,
        "execution": summary,
        "reran_cells": False,
    }
    manifest_path = destination_root / f"{archived_path.stem}.json"
    write_json_atomic(manifest_path, manifest)
    result = {**manifest, "manifest_path": str(manifest_path)}
    if require_complete and summary["status"] != "complete":
        raise NotebookExecutionError(
            f"Archived notebook execution status is {summary['status']}: {manifest_path}"
        )
    return result
