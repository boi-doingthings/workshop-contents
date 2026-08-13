#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PROJECT_DIR}/scripts/container.sh" bash -lc '
set -euo pipefail
source .venv/bin/activate
exec python -m jupyterlab --ip=0.0.0.0 --port="${JUPYTER_PORT:-8888}" --no-browser \
  --ServerApp.root_dir=/workspace --ServerApp.allow_remote_access=true
'
