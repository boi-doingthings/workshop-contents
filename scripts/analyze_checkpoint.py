#!/usr/bin/env python3
"""Analyze BF16 error and scales from one real packed ModelOpt checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ptq_workshop.artifacts import write_derived_json_atomic
from ptq_workshop.checkpoint_analysis import analyze_packed_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp8", "nvfp4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor", action="append", dest="tensors")
    parser.add_argument("--tensor-count", type=int, default=3)
    parser.add_argument("--maximum-samples", type=int, default=1_000_000)
    args = parser.parse_args()
    report = analyze_packed_checkpoint(
        args.source,
        args.checkpoint,
        variant=args.precision,
        tensor_names=args.tensors,
        tensor_count=args.tensor_count,
        maximum_samples_per_tensor=args.maximum_samples,
    )
    write_derived_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
