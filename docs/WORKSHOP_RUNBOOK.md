# Single-B200/B300 workshop runbook

## Before the session

1. Confirm the selected B200 or B300 is idle, healthy, and visible to Docker.
2. Inspect the operation-scoped storage plan: 70 GiB only when the pinned BF16 snapshot is absent,
   2 GiB for cached preparation/results-only work, and 55 GiB for a fresh FP8+NVFP4 export pair.
3. Run `./scripts/bootstrap.sh` and `./scripts/prepare_assets.sh WORKSHOP_B200`.
4. Run the `DEV_SMOKE` profile or an abbreviated target-GPU smoke pass after any driver/container change.
5. Choose `--resume` to validate/reuse the matching packed exports or `--fresh` to reset the exact
   matching instructor run once. Never copy checkpoints or result files between fingerprints.

The preparation manifest must say `"download_only": true`. The instructor should retain a separate
completed rehearsal run for emergency screenshots, clearly labeled **rehearsal**, but it is not an
input to the live run.

## Live sequence

1. Launch `./scripts/launch_notebook.sh --profile WORKSHOP_B200 --live --fresh --port 8888`, open
   the tokenized loopback URL through SSH forwarding, and select **Blackwell PTQ (pinned)**.
2. Run the setup cell once; it consumes the fresh-reset control and displays the
   GPU/environment fingerprint plus physical, reusable, and reclaimable storage.
3. Teach numeric formats and use the small synthetic quantization visualization.
4. Inspect the Nemotron architecture, theoretical size, and high-precision exclusions.
5. Materialize/check the frozen calibration manifest and run FP8 PTQ/export.
6. Run NVFP4 PTQ/export from BF16 and validate both packed checkpoints.
7. Run the `analyze` stage. Confirm its manifests use ModelOpt QTensor dequantizers and contain
   representative BF16 error CDFs and weight-scale distributions for both packed formats.
8. Run fixed accuracy subsets, then the three benchmark scenarios with NVML telemetry.
9. Generate the dashboard and discuss the measured quality/performance/memory/energy Pareto frontier.
10. Save the notebook, then run `./scripts/container.sh bash -lc 'source .venv/bin/activate &&
    python scripts/archive_notebook.py --run-dir <run-dir>'`; retain the hash-addressed notebook and
    execution manifest with the run.

## Result validity gates

- The manifest identifies exactly one B200 or B300, records compute capability 10.x, and uses TP=1.
- Both quantized artifacts cite the pinned BF16 revision and have identical exclusions.
- Headline variants report no KV quantization.
- All scale probes are finite and deterministic smoke responses are nonempty.
- PTQ observability manifests contain total/calibration/export durations and peak VRAM.
- Parsed-answer rate is at least 98%; a quality loss above five percentage points is prominent.
- Measured runs have full telemetry coverage and no throttling/external utilization warning.
- Repetitions above 5% coefficient of variation are extended to five or labeled unstable.

If a gate fails, keep its logs and teach from the last completed valid stage. Never replace real
NVFP4 timings with simulated-quantization timings or an official checkpoint.
