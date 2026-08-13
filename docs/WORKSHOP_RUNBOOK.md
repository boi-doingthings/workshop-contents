# Single-B200 workshop runbook

## Before the session

1. Confirm the selected B200 is idle, healthy, and visible to Docker.
2. Confirm at least 145 GiB is free on the persistent model/artifact volume.
3. Run `./scripts/bootstrap.sh` and `./scripts/prepare_assets.sh WORKSHOP_B200`.
4. Run the `DEV_SMOKE` profile or an abbreviated B200 smoke pass after any driver/container change.
5. Do not generate or copy FP8/NVFP4 checkpoints or result files into the workshop run directory.

The preparation manifest must say `"download_only": true`. The instructor should retain a separate
completed rehearsal run for emergency screenshots, clearly labeled **rehearsal**, but it is not an
input to the live run.

## Live sequence

1. Open the notebook and select **Blackwell PTQ (pinned)**.
2. Select `WORKSHOP_B200`; create the new run and display its GPU/environment fingerprint.
3. Teach numeric formats and use the small synthetic quantization visualization.
4. Inspect the Nemotron architecture, theoretical size, and high-precision exclusions.
5. Materialize/check the frozen calibration manifest and run FP8 PTQ/export.
6. Run NVFP4 PTQ/export from BF16 and validate both packed checkpoints.
7. Run the `analyze` stage. Confirm its manifests use ModelOpt QTensor dequantizers and contain
   representative BF16 error CDFs and weight-scale distributions for both packed formats.
8. Run fixed accuracy subsets, then the three benchmark scenarios with NVML telemetry.
9. Generate the dashboard and discuss the measured quality/performance/memory/energy Pareto frontier.
10. Save the notebook, then run `.venv/bin/python scripts/archive_notebook.py --run-dir <run-dir>`; retain the
    hash-addressed notebook and execution manifest with the run.

## Result validity gates

- The manifest identifies a single B200 and TP=1.
- Both quantized artifacts cite the pinned BF16 revision and have identical exclusions.
- Headline variants report no KV quantization.
- All scale probes are finite and deterministic smoke responses are nonempty.
- PTQ observability manifests contain total/calibration/export durations and peak VRAM.
- Parsed-answer rate is at least 98%; a quality loss above five percentage points is prominent.
- Measured runs have full telemetry coverage and no throttling/external utilization warning.
- Repetitions above 5% coefficient of variation are extended to five or labeled unstable.

If a gate fails, keep its logs and teach from the last completed valid stage. Never replace real
NVFP4 timings with simulated-quantization timings or an official checkpoint.
