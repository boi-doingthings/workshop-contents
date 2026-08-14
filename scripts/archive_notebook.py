#!/usr/bin/env python3
"""Archive a saved, already-executed notebook into an immutable run artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ptq_workshop.notebook_artifacts import archive_executed_notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "notebooks"
        / "blackwell_ptq_workshop.ipynb",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return success after archiving a partial/failed notebook.",
    )
    args = parser.parse_args()
    result = archive_executed_notebook(
        args.notebook,
        args.run_dir,
        require_complete=not args.allow_partial,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
