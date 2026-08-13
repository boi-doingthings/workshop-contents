#!/usr/bin/env python3
"""Evaluate one precision through the canonical deterministic harness."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp8", "nvfp4"), required=True)
    parser.add_argument("--task", choices=("all", "mmlu_pro", "gsm8k"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_profile.py")),
        "--profile",
        args.profile,
        "--run-dir",
        str(args.run_dir),
        "--stage",
        "evaluate",
        "--precision",
        args.precision,
        "--task",
        args.task,
    ]
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
