#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${PTQ_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release@sha256:998068efffcddb06905b83e9e712a4aec9f39d8f1ec4afacf6c0f3bac4479b54}"
GPU_ID="${PTQ_GPU_ID:-0}"
HF_CACHE="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
mkdir -p "${HF_CACHE}" "${PROJECT_DIR}/artifacts" "${PROJECT_DIR}/.cache/home" "${PROJECT_DIR}/.cache/torchinductor"

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

exec docker run --rm "${TTY_ARGS[@]}" \
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
  --volume "${PROJECT_DIR}:/workspace" \
  --volume "${HF_CACHE}:/workspace/.cache/huggingface" \
  "${IMAGE}" "$@"
