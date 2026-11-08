---
name: wakeword-training-cycle
description: Use when training, retraining or improving a wake-word model with wakeword-lab (prepare, experiment, sweep, evaluate, report), when interpreting run reports or metrics (recall, FPR, threshold), or when deciding what to change after a bad run (low recall, false positives, stuck validation, threshold miscalibration).
---

# Wake-word training cycle

```bash
wakeword-lab prepare <proj>                                            # after any data change
wakeword-lab experiment <proj> --run-name <name> [--width 24 --dropout 0.2]
```

Every run writes `artifacts/experiments/<name>/report/` (report.md + plots) and a summary JSON. Read the report before changing anything.

## Read the report in this order

1. **`report.md` metrics table** — validation vs test recall/FP at the calibrated threshold.
2. **`score_distributions.png`** — positive vs negative histograms with the threshold line. The single most diagnostic plot.
3. **`training_curves.png`** — val metrics should climb with train metrics; a flat val line = generalization failure.
4. **`roc_pr.png` / `quantization_drift.png`** — threshold headroom; int8 drift (MAE < 0.01 is fine).

## Symptom → lever

| Symptom | Diagnosis | Lever |
|---|---|---|
| Train acc ~99%, val acc stuck ~35% | TTS overfit, model never saw real voice | Real train recordings (see wakeword-data-collection) |
| Val recall high in training, 0 at calibrated threshold | Too few validation negatives → threshold = max neg score | More validation negatives |
| Positives score ~0.5, threshold slightly above | Weak margin | More/better data first; then dropout/width sweep |
| Negatives median far above 0 | Too few/too easy training negatives | fetch-negatives --count 800+, near-miss TTS negatives, real negatives |
| Recall drops only on noise-mixed copies | Expected at 0 dB SNR; partially fine | `noise_mixing.copies_per_clip: 2` if robustness is the goal |
| Test FP > 0 on specific words | Hard negatives in test | Record those sounds as negative/train |
| Metrics worse after more data | Check split leakage first (manifest groups per split) | Re-check metadata pins; then sweep |

## Knobs

- `--width` / `--dropout`: more dropout (0.2) helps when data is small. `wakeword-lab sweep <proj>` + `wakeword-lab compare <proj>` ranks configs by validation (never test).
- `wakeword.yaml`: augmentation copies, noise_mixing SNR range/copies, quantization.target_validation_fpr.

## Test-set discipline

Validation drives all decisions (threshold calibration, sweeps, comparisons). Test is a one-time check. Iterating against test inflates results — after several test evaluations, collect a fresh test session before claiming final numbers.

## Iteration loop

data change → prepare → experiment (new --run-name) → compare report.md tables → keep the better run. Never overwrite a good run's directory; run names are cheap.
