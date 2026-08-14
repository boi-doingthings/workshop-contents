# Blackwell PTQ Workshop

An instructor-led, reproducible comparison of the pinned
[`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
checkpoint in BF16, ModelOpt FP8, and ModelOpt NVFP4 on NVIDIA Blackwell.

The project deliberately separates Model Optimizer's simulated calibration from real deployment.
Accuracy and performance are measured only after export to packed unified Hugging Face checkpoints
and reload through the same TensorRT-LLM runtime.

## What is pinned

- Model revision: `2d59de1cbd51c0adf384eb906b766d1aee0e0517`
- TensorRT-LLM: `1.3.0rc23`, image digest
  `sha256:316b840a08a8174fc3f6b5716828bdfe1daaf629ee1ac2a8b7a22526d141a007`
- NVIDIA Model Optimizer: official tag `0.46.0rc0`, commit
  `33d05b0c446f528914173041057050f6d135fbf4`
- Calibration corpus: deterministic, hashed CNN/DailyMail rows
- Seed: `42`
- Primary deployment: one B200 or B300 Blackwell GPU, TP=1, BF16 KV cache

The exact resolved Python environment is written to `pip-freeze.txt` by bootstrap and copied into
every run manifest. Credentials are never written to artifacts.

`0.46.0rc0` is a controlled compatibility exception to the usual stable-release preference.
ModelOpt `0.45.0` deliberately backported NemotronH non-gated fused-MoE calibration without the
unified Hugging Face export path; a live FP8 run therefore calibrated successfully and then failed
with `NotImplementedError: MoE model with experts type 'QuantNemotronHExperts' is not supported in
export`. NVIDIA's complete fix is commit
[`c81210fa`](https://github.com/NVIDIA/Model-Optimizer/commit/c81210faecc096a7bd802cca2cda909ac43f7759),
which is contained in the first official containing tag, `0.46.0rc0`. Bootstrap clones that tag,
verifies the exact commit above, and installs from the verified source tree; downgrading to `0.45.0`
would make this Nemotron checkpoint non-exportable.

## Quick start

The host needs Linux, Docker with NVIDIA Container Toolkit, and an accessible Blackwell GPU. Disk
checks are operation-scoped: an absent BF16 snapshot needs 70 GiB for download, a verified cached
snapshot needs only 2 GiB to refresh profile-specific datasets, and a fresh FP8+NVFP4 export pair
needs 55 GiB. Existing valid packed checkpoints are credited on resume; `--fresh` credits only the
exact run directory it will replace. B200 and B300 both use the
`WORKSHOP_B200` profile (the profile name records the original workshop target); RTX PRO 6000 uses
`DEV_SMOKE` for development validation. The model is public; `HF_TOKEN` is optional but recommended
to avoid Hub rate limits.

```bash
git clone https://github.com/boi-doingthings/workshop-contents.git
cd workshop-contents
./scripts/bootstrap.sh
./scripts/prepare_assets.sh DEV_SMOKE       # downloads only; no PTQ or results
./scripts/launch_notebook.sh --profile DEV_SMOKE --dry-run --resume --port 8888
```

Open `notebooks/blackwell_ptq_workshop.ipynb` with the **Blackwell PTQ (pinned)** kernel.

For the instructor's live RTX run, use:

```bash
# Reuse valid quantized checkpoints, but regenerate requested metrics/predictions.
./scripts/launch_notebook.sh --profile DEV_SMOKE --live --resume --port 8888

# Or replace the one matching run from the setup cell, then execute every PTQ step live.
./scripts/launch_notebook.sh --profile DEV_SMOKE --live --fresh --port 8888
```

Jupyter binds only to the machine's loopback interface and prints a tokenized URL. From a local
workstation, forward it with `ssh -N -L 8888:127.0.0.1:8888 <user>@<host>`. Code Server on port
8080 remains a convenient artifact browser/editor; notebook execution stays in Jupyter's pinned
container kernel. Forward both ports if you want both interfaces.

The `.venv` is deliberately created inside the pinned container with Python 3.12 and
`--system-site-packages`, so it can inherit the image's CUDA, PyTorch, TensorRT-LLM, and Transformers
stack. Do not run `.venv/bin/python` directly on the host or install packages into it from the host;
execute workshop commands through `scripts/container.sh` as shown below.

For the B200 or B300 rehearsal (the historical profile name remains `WORKSHOP_B200`):

```bash
PTQ_GPU_ID=0 ./scripts/prepare_assets.sh WORKSHOP_B200
PTQ_GPU_ID=0 ./scripts/container.sh bash -lc \
  'source .venv/bin/activate && python scripts/run_profile.py --profile WORKSHOP_B200 --stage all'
```

Preparation downloads only the BF16 source and frozen datasets. The instructor notebook uses a
deterministic run fingerprint and performs quantization, validation, accuracy, performance,
telemetry, and reporting live; `--fresh` resets that matching run once, while `--resume` validates
and reuses its exports. Fresh PTQ stages also persist total wall time, ModelOpt-reported calibration/export
durations, raw NVML samples, and peak VRAM. The analysis stage compares representative BF16 source
weights with the corresponding packed FP8/NVFP4 tensors dequantized by ModelOpt itself.

## Profiles

| Profile | Calibration | Accuracy subset | Performance repeats | Purpose |
|---|---:|---:|---:|---|
| `DEV_SMOKE` | 16 | small fixed MMLU-Pro/GSM8K | 1 shortened repeat | End-to-end RTX PRO 6000 development gate |
| `WORKSHOP_B200` | 128 | 300 MMLU-Pro + 100 GSM8K | 3, up to 5 on CV >5% | 90-minute single-B200/B300 instructor run |
| `FULL` | 128 | 1,000 MMLU-Pro + 250 GSM8K | 3, up to 5 | Post-workshop analysis |

Every result records its GPU UUID and model. RTX PRO 6000 development measurements are never
presented as B200 or B300 results, and results from one data-center SKU are never relabeled as the
other.

## Experiment controls

- FP8 and NVFP4 always start from the same pinned BF16 snapshot.
- Their high-precision exclusion lists are identical: LM head, Mamba Conv1d, all attention Q/K/V/O
  projections, and the Mamba projections immediately preceding attention stay BF16.
- Recipe selectors use Transformers' live `model.layers.*` quantizer namespace. Validation checks
  both ModelOpt's `.quant_summary.txt` (all 72 controlled input/weight quantizers disabled) and the
  36 serialized `backbone.layers.*` weight tensors directly (all BF16); the translated export
  `exclude_modules` list is not treated as sufficient proof.
- ModelOpt 0.46.0rc0 unified-HF export serializes Nemotron-H's derived `layers_block_type` property,
  but the copied config class exposes that property without a setter. Immediately after export the
  wrapper atomically removes only that field, preserves `mtp_layers_block_type` and the complete
  embedded `quantization_config`, and records the removed value plus before/after SHA-256 hashes in
  checkpoint-local `.config_normalization.json`. Validation rejects an unnormalized checkpoint.
- The same exporter can embed NUL-delimited internal name sentinels in both ModelOpt exclusion
  lists. A second atomic, hash-chained normalization replaces those lists with NVIDIA's exact
  published 60-module order (LM head, 36 sensitive projections, and 23 Mamba convolutions), proves
  every entry has a serialized `.weight` tensor, aligns `config.json` with `hf_quant_config.json`,
  and declares `torch_dtype: bfloat16` for the BF16 fallback modules. Its original/replacement list
  hashes and both files' before/after hashes live in
  `.quantization_metadata_normalization.json`; reruns verify and reuse both audits byte-for-byte.
- Unified-HF export also omits Nemotron-H's two raw scheduling inputs even though the copied config
  uses them to derive `layers_block_type`. A final hash-chained normalization restores only
  `hybrid_override_pattern` and `num_hidden_layers` from the pinned BF16 source, checks the exact
  52-layer Mamba/MoE/attention schedule, and proves the safetensors index metadata did not change.
  `.source_topology_normalization.json` binds the source config SHA-256, both restored value hashes,
  the config transition, and unchanged tensor metadata; validation and runtime freshness require it.
- KV cache stays BF16 in the headline comparison. FP8 KV is an optional, separate extension.
- Evaluation uses reasoning-off, greedy decoding and identical prompt IDs.
- Performance uses fixed token lengths, TP=1, no speculative decoding or prefix reuse, one clean
  model process per precision, warm-up, and repeated measurements.
- Any changed scenario, OOM fallback, missing native kernel, unstable run, or accuracy regression is
  recorded and surfaced; it is never silently normalized away.

NVIDIA's published Nemotron 3 Nano NVFP4 model received quantization-aware distillation after PTQ.
Its published quality is displayed only as an external reference, not as this notebook's pure-PTQ
result. See the [official NVFP4 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4).

## Artifacts

Each `artifacts/runs/<run-id>/` contains immutable configuration manifests, refreshed environment
observations, resolved
recipes, PTQ logs, packed checkpoints, raw predictions, benchmark JSON, NVML CSV, figures,
`summary.csv`, and `summary.json`. The report is regenerated solely from those machine-readable
files. The instructor notebook uses one deterministic run per experiment fingerprint. Resume mode
validates and reuses packed checkpoints while replacing retryable predictions, telemetry and
reports. Fresh mode performs one guarded whole-run reset after verifying the manifest and projected
free-plus-reclaimable capacity; it never deletes unrelated runs or overwrites checkpoint files in
place. Frozen inputs live under `artifacts/prepared/<profile>/`, preventing a DEV_SMOKE subset from
being consumed accidentally by WORKSHOP_B200.

The `manifests/<precision>-ptq-observability.json` files retain quantization wall time, parsed
calibration/export timing, peak VRAM, and links to raw quantization telemetry. The
`manifests/<precision>-tensor-analysis.json` files retain real BF16-versus-dequantized error metrics,
CDF points, and scale distributions. A parsed-answer rate below 98% writes
`metrics/accuracy-acceptance.json` and then fails the stage; raw predictions and score reports stay
in place for diagnosis.

After saving the executed workshop notebook, archive it without rerunning any cells:

```bash
./scripts/container.sh bash -lc \
  'source .venv/bin/activate && python scripts/archive_notebook.py --run-dir artifacts/runs/<run-id>'
```

This creates a hash-addressed notebook and execution manifest in `<run-id>/notebooks/`. Partial or
failed notebooks are copied before the command reports failure, preserving the evidence.

Every serving attempt also writes a retryable
`metrics/<precision>-<evaluate|benchmark|all>-runtime-status.json`. Startup, exact-sentinel smoke,
evaluation, or benchmark failures are marked unavailable with the stage, exception, observed smoke
text, attempt timestamp, and stage-specific server log. Reports retain checkpoint-footprint facts
but suppress stale accuracy, performance, and telemetry for an unavailable runtime.

## Troubleshooting

- **Preflight rejects the GPU:** NVFP4 is native only on Blackwell. B200 reports compute capability
  10.0, B300 reports 10.3, and RTX PRO 6000 Blackwell reports 12.0. The runtime accepts data-center
  Blackwell 10.x and RTX Blackwell 12.0; DGX Spark 12.1 and RTX 6000 Ada 8.9 are intentionally out
  of scope. See NVIDIA's [CUDA GPU compute-capability table](https://developer.nvidia.com/cuda/gpus).
- **Disk check fails:** inspect the recorded `storage_plan`. Missing source, cached preparation,
  fresh exports, and results-only resumes have separate budgets. Point `HF_HOME` at a persistent
  volume if the 70 GiB source-download budget is unavailable. The project does not delete unrelated
  data.
- **PTQ OOM:** `DEV_SMOKE` may retry the documented ModelOpt low-memory path when compatible.
  Workshop dimensions and formats are never silently changed.
- **NVFP4 startup fails:** inspect the saved runtime-status JSON and linked server log. All Blackwell
  targets use AutoDeploy with real packed NVFP4 checkpoints. B200/B300 use the `trtllm_gen` MoE
  kernel from the normal Nano-v3 configuration. RTX PRO 6000 (SM120) uses the dedicated
  `nano_v3_sm120.yaml`: its MoE stays real NVFP4 through CUTLASS because rc23's `trtllm_gen`
  batched-MoE runner rejects SM120. The fused RMSNorm quantizer rejects this model's activation
  shape, and the fused ReLU² quantizer is disabled because its controlled diagnostic exceeded the
  five-point accuracy-regression guard. The underlying NVFP4 linear and MoE operations remain quantized.
  The exact-generation gate prevents benchmark or accuracy numbers from being reported if this
  hardware-specific route ever regresses.
- **Hub throttling:** export `HF_TOKEN`; manifests record only a Boolean indicating its presence.

## Primary references

- [NVIDIA Model Optimizer Hugging Face PTQ example](https://github.com/NVIDIA/Model-Optimizer/tree/0.46.0rc0/examples/hf_ptq)
- [ModelOpt quantization guide](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)
- [Unified Hugging Face checkpoint deployment](https://nvidia.github.io/Model-Optimizer/deployment/3_unified_hf.html)
- [TensorRT-LLM quantization support](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)
- [TensorRT-LLM 1.3.0rc23 release notes](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc23)
- [TensorRT-LLM Nemotron-3 deployment guide](https://nvidia.github.io/TensorRT-LLM/latest/deployment-guide/deployment-guide-for-nemotron-3-on-trtllm.html)
- [TensorRT-LLM SM120/SM121 NVFP4 MoE implementation](https://github.com/NVIDIA/TensorRT-LLM/pull/13773)
- [TensorRT-LLM benchmarking](https://nvidia.github.io/TensorRT-LLM/1.3.0rc23/commands/trtllm-bench.html)
- [NVIDIA NVFP4 format](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [NVIDIA CUDA GPU compute capability table](https://developer.nvidia.com/cuda/gpus)
