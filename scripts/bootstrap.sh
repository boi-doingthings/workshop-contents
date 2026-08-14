#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${PROJECT_DIR}/scripts/container.sh" bash -lc '
set -euo pipefail
mkdir -p "$HOME" .cache
MODELOPT_TAG="0.46.0rc0"
MODELOPT_COMMIT="33d05b0c446f528914173041057050f6d135fbf4"
MODELOPT_DIR=".cache/Model-Optimizer-0.46.0rc0"
if [[ ! -d "${MODELOPT_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${MODELOPT_TAG}" https://github.com/NVIDIA/Model-Optimizer.git "${MODELOPT_DIR}"
fi
ACTUAL_MODELOPT_COMMIT="$(git -C "${MODELOPT_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_MODELOPT_COMMIT}" != "${MODELOPT_COMMIT}" ]]; then
  echo "ModelOpt checkout mismatch: got ${ACTUAL_MODELOPT_COMMIT}; required ${MODELOPT_COMMIT}" >&2
  exit 1
fi
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
# Controlled exception: 0.45.0 intentionally omitted NemotronH unified-HF
# export. Install the exact first NVIDIA tag carrying the complete fix from its
# verified source checkout. --no-deps preserves the validated TRT-LLM image stack.
# Keep this non-editable. An editable ModelOpt source install leaves its namespace
# visible beside the image system ModelOpt distribution and can mix modules
# across releases (observed as a config_loader import failure).
python -m pip install --no-deps --force-reinstall "${MODELOPT_DIR}"
python -m pip install "pytest==8.4.1"
python -m pip install --no-deps -e .
python -m ipykernel install --user --name blackwell-ptq --display-name "Blackwell PTQ (pinned)"
python -m pip freeze | sort > pip-freeze.txt
python - <<"PY"
import datasets, huggingface_hub, inspect, modelopt, torch, transformers, tensorrt_llm, yaml
import jupyterlab, matplotlib, nbclient, nbformat, numpy, openai, pandas, pynvml, scipy
import setuptools
from modelopt.recipe import load_recipe
from modelopt.torch.export import unified_export_hf
from modelopt.torch.opt import config_loader
from modelopt.torch.quantization.plugins import huggingface as modelopt_hf_plugin
from ptq_workshop.preflight import is_supported_compute_capability
assert torch.cuda.is_available(), "CUDA is not visible inside the pinned container"
capability = torch.cuda.get_device_capability(0)
assert is_supported_compute_capability(capability), (
    f"Unsupported GPU compute capability {capability}; "
    "requires B200/B300/GB200/GB300 10.x or RTX Blackwell 12.0"
)
assert transformers.__version__ == "5.5.4"
assert tensorrt_llm.__version__ == "1.3.0rc23"
assert datasets.__version__ == "3.1.0"
assert huggingface_hub.__version__ == "1.14.0"
assert setuptools.__version__ == "79.0.1"
assert modelopt.__version__ == "0.46.0rc0"
venv_site = "/workspace/.venv/lib/python3.12/site-packages/modelopt/"
for imported in (modelopt, config_loader, unified_export_hf, modelopt_hf_plugin):
    imported_path = inspect.getfile(imported)
    assert imported_path.startswith(venv_site), (
        f"Mixed ModelOpt namespace: {imported.__name__} loaded from {imported_path}"
    )
load_recipe("configs/recipes/fp8.yaml")
load_recipe("configs/recipes/nvfp4.yaml")
print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("TensorRT-LLM:", tensorrt_llm.__version__)
print("ModelOpt:", modelopt.__version__)
print("Transformers:", transformers.__version__)
PY
'

echo "Environment ready. Launch with: ${PROJECT_DIR}/scripts/launch_notebook.sh"
