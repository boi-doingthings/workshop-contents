#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${PTQ_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release@sha256:316b840a08a8174fc3f6b5716828bdfe1daaf629ee1ac2a8b7a22526d141a007}"
GPU_ID="${PTQ_GPU_ID:-0}"
HF_CACHE="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
mkdir -p "${HF_CACHE}" "${PROJECT_DIR}/artifacts" "${PROJECT_DIR}/.cache/home" "${PROJECT_DIR}/.cache/torchinductor"

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

PTQ_PROFILE_VALUE="${PTQ_PROFILE:-DEV_SMOKE}"
PTQ_DRY_RUN_VALUE="${PTQ_DRY_RUN:-1}"
JUPYTER_PORT_VALUE="${JUPYTER_PORT:-8888}"
if [[ ! "${PTQ_PROFILE_VALUE}" =~ ^(DEV_SMOKE|WORKSHOP_B200|FULL)$ ]]; then
  echo "Invalid PTQ_PROFILE: ${PTQ_PROFILE_VALUE}" >&2
  exit 2
fi
if [[ ! "${PTQ_DRY_RUN_VALUE}" =~ ^[01]$ ]]; then
  echo "PTQ_DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${PTQ_START_FRESH:-0}" =~ ^[01]$ ]]; then
  echo "PTQ_START_FRESH must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${JUPYTER_PORT_VALUE}" =~ ^[0-9]+$ ]] || (( 10#${JUPYTER_PORT_VALUE} < 1 || 10#${JUPYTER_PORT_VALUE} > 65535 )); then
  echo "JUPYTER_PORT must be an integer in [1, 65535]" >&2
  exit 2
fi
PORT_ARGS=()
if [[ "${PTQ_PUBLISH_JUPYTER:-0}" == "1" ]]; then
  PORT_ARGS=(--publish "127.0.0.1:${JUPYTER_PORT_VALUE}:${JUPYTER_PORT_VALUE}")
fi

exec docker run --rm "${TTY_ARGS[@]}" "${PORT_ARGS[@]}" \
  --gpus "device=${GPU_ID}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  --workdir /workspace \
  --env HOME=/workspace/.cache/home \
  --env USER=workshop \
  --env LOGNAME=workshop \
  --env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env PYTHONUNBUFFERED=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env PTQ_PROFILE="${PTQ_PROFILE_VALUE}" \
  --env PTQ_DRY_RUN="${PTQ_DRY_RUN_VALUE}" \
  --env PTQ_START_FRESH="${PTQ_START_FRESH:-0}" \
  --env JUPYTER_PORT="${JUPYTER_PORT_VALUE}" \
  --volume "${PROJECT_DIR}:/workspace" \
  --volume "${HF_CACHE}:/workspace/.cache/huggingface" \
  "${IMAGE}" "$@"
