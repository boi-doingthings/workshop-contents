# Blackwell PTQ Workshop

An instructor-led, reproducible comparison of the pinned
[`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
checkpoint in BF16, ModelOpt FP8, and ModelOpt NVFP4 on NVIDIA Blackwell.

The project deliberately separates Model Optimizer's simulated calibration from real deployment.
Accuracy and performance are measured only after export to packed unified Hugging Face checkpoints
and reload through the same TensorRT-LLM runtime.

## What is pinned

- Model revision: `2d59de1cbd51c0adf384eb906b766d1aee0e0517`
- TensorRT-LLM: `1.3.0rc17`, image digest
  `sha256:998068efffcddb06905b83e9e712a4aec9f39d8f1ec4afacf6c0f3bac4479b54`
- NVIDIA Model Optimizer: official tag `0.46.0rc0`, commit
  `33d05b0c446f528914173041057050f6d135fbf4`
- Calibration corpus: deterministic, hashed CNN/DailyMail rows
- Seed: `42`
- Primary deployment: one Blackwell GPU, TP=1, BF16 KV cache

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

The host needs Linux, Docker with NVIDIA Container Toolkit, an accessible Blackwell GPU, and at
least 145 GiB free in the project/cache filesystem. B200 and B300 both use the
`WORKSHOP_B200` profile (the profile name records the original workshop target); RTX PRO 6000 uses
`DEV_SMOKE` for development validation. The model is public; `HF_TOKEN` is optional but recommended
to avoid Hub rate limits.

```bash
git clone https://github.com/boi-doingthings/workshop-contents.git
cd workshop-contents
./scripts/bootstrap.sh
./scripts/prepare_assets.sh DEV_SMOKE       # downloads only; no PTQ or results
./scripts/launch_notebook.sh
```

Open `notebooks/blackwell_ptq_workshop.ipynb` with the **Blackwell PTQ (pinned)** kernel.

The `.venv` is deliberately created inside the pinned container with Python 3.12 and
`--system-site-packages`, so it can inherit the image's CUDA, PyTorch, TensorRT-LLM, and Transformers
stack. Do not run `.venv/bin/python` directly on the host or install packages into it from the host;
execute workshop commands through `scripts/container.sh` as shown below.

For the B200 rehearsal:

```bash
PTQ_GPU_ID=0 ./scripts/prepare_assets.sh WORKSHOP_B200
PTQ_GPU_ID=0 ./scripts/container.sh bash -lc \
  'source .venv/bin/activate && python scripts/run_profile.py --profile WORKSHOP_B200 --stage all'
```

Preparation downloads only the BF16 source and frozen datasets. `WORKSHOP_B200` creates a fresh,
timestamped run and performs quantization, validation, accuracy, performance, telemetry, and
reporting live. Fresh PTQ stages also persist total wall time, ModelOpt-reported calibration/export
durations, raw NVML samples, and peak VRAM. The analysis stage compares representative BF16 source
weights with the corresponding packed FP8/NVFP4 tensors dequantized by ModelOpt itself.

## Profiles

| Profile | Calibration | Accuracy subset | Performance repeats | Purpose |
|---|---:|---:|---:|---|
| `DEV_SMOKE` | 16 | small fixed MMLU-Pro/GSM8K | 1 shortened repeat | End-to-end RTX PRO 6000 development gate |
| `WORKSHOP_B200` | 128 | 300 MMLU-Pro + 100 GSM8K | 3, up to 5 on CV >5% | 90-minute single-B200 instructor run |
| `FULL` | 128 | 1,000 MMLU-Pro + 250 GSM8K | 3, up to 5 | Post-workshop analysis |

Every result records its GPU UUID and model. RTX PRO 6000 development measurements are never
presented as B200 results.

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

Each `artifacts/runs/<run-id>/` contains immutable configuration/environment manifests, resolved
recipes, PTQ logs, packed checkpoints, raw predictions, benchmark JSON, NVML CSV, figures,
`summary.csv`, and `summary.json`. The report is regenerated solely from those machine-readable
files. Matching fingerprints are resumable for development; workshop runs intentionally receive a
new ID.

The `manifests/<precision>-ptq-observability.json` files retain quantization wall time, parsed
calibration/export timing, peak VRAM, and links to raw quantization telemetry. The
`manifests/<precision>-tensor-analysis.json` files retain real BF16-versus-dequantized error metrics,
CDF points, and scale distributions. A parsed-answer rate below 98% writes
`metrics/accuracy-acceptance.json` and then fails the stage; raw predictions and score reports stay
in place for diagnosis.

After saving the executed workshop notebook, archive it without rerunning any cells:

```bash
.venv/bin/python scripts/archive_notebook.py --run-dir artifacts/runs/<run-id>
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
  10.x; RTX PRO 6000 Blackwell reports 12.0. RTX 6000 Ada is unsupported.
- **Disk gate fails:** point `HF_HOME` at a persistent volume with enough space before bootstrap.
  The project does not delete unrelated data.
- **PTQ OOM:** `DEV_SMOKE` may retry the documented ModelOpt low-memory path when compatible.
  Workshop dimensions and formats are never silently changed.
- **No NVFP4 performance:** a valid export is retained, but fake-quant timing is never substituted
  for missing real runtime kernels.
- **Hub throttling:** export `HF_TOKEN`; manifests record only a Boolean indicating its presence.

## Primary references

- [NVIDIA Model Optimizer Hugging Face PTQ example](https://github.com/NVIDIA/Model-Optimizer/tree/0.46.0rc0/examples/hf_ptq)
- [ModelOpt quantization guide](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)
- [Unified Hugging Face checkpoint deployment](https://nvidia.github.io/Model-Optimizer/deployment/3_unified_hf.html)
- [TensorRT-LLM quantization support](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)
- [TensorRT-LLM benchmarking](https://nvidia.github.io/TensorRT-LLM/1.3.0rc21/commands/trtllm-bench.html)
- [NVIDIA NVFP4 format](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
