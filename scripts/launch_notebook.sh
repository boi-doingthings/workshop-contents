#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${PTQ_PROFILE:-DEV_SMOKE}"
DRY_RUN="${PTQ_DRY_RUN:-1}"
START_FRESH="${PTQ_START_FRESH:-0}"
PORT="${JUPYTER_PORT:-8888}"

while (($#)); do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { echo "--profile requires a value" >&2; exit 2; }
      PROFILE="$2"; shift 2 ;;
    --live) DRY_RUN=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --fresh) START_FRESH=1; shift ;;
    --resume) START_FRESH=0; shift ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a value" >&2; exit 2; }
      PORT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--profile DEV_SMOKE|WORKSHOP_B200|FULL] [--live|--dry-run] [--fresh|--resume] [--port PORT]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

export PTQ_PROFILE="${PROFILE}"
export PTQ_DRY_RUN="${DRY_RUN}"
export PTQ_START_FRESH="${START_FRESH}"
export JUPYTER_PORT="${PORT}"
export PTQ_PUBLISH_JUPYTER=1

echo "Launching Blackwell PTQ JupyterLab: profile=${PROFILE} dry_run=${DRY_RUN} fresh=${START_FRESH}"
echo "Local-only endpoint: http://127.0.0.1:${PORT}/lab (use SSH port forwarding remotely)"
exec "${PROJECT_DIR}/scripts/container.sh" bash -lc '
set -euo pipefail
source .venv/bin/activate
exec python -m jupyterlab --ip=0.0.0.0 --port="${JUPYTER_PORT:-8888}" --no-browser \
  --ServerApp.root_dir=/workspace --ServerApp.allow_remote_access=true
'
