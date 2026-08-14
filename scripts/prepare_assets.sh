#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-WORKSHOP_B200}"
exec "${PROJECT_DIR}/scripts/container.sh" bash -lc "
set -euo pipefail
source .venv/bin/activate
python scripts/prepare_assets.py --profile '${PROFILE}'
"
