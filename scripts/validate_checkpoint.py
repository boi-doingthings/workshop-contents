#!/usr/bin/env python3
"""Validate a BF16 source or packed ModelOpt unified-HF checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ptq_workshop.io import atomic_write_json
from ptq_workshop.quantize import validate_export


def validate_bf16(path: Path) -> dict[str, object]:
    required = ("config.json", "model.safetensors.index.json")
    missing = [name for name in required if not (path / name).is_file()]
    tensors = sorted(path.glob("*.safetensors"))
    if missing or not tensors:
        details = [*(f"missing {name}" for name in missing)]
        if not tensors:
            details.append("no .safetensors files")
        raise RuntimeError(f"Invalid BF16 source {path}: {', '.join(details)}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if str(config.get("torch_dtype", config.get("dtype", ""))).lower() not in {
        "bfloat16",
        "bf16",
    }:
        raise RuntimeError(f"Source config does not declare BF16 weights: {path}")
    return {
        "precision": "bf16",
        "path": str(path.resolve()),
        "files": len(tensors),
        "bytes": sum(item.stat().st_size for item in tensors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--precision", choices=("bf16", "fp8", "nvfp4"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    result = (
        validate_bf16(checkpoint)
        if args.precision == "bf16"
        else validate_export(checkpoint, expected_variant=args.precision)
    )
    result["precision"] = args.precision
    if args.output:
        atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
