---
name: wakeword-project-setup
description: Use when a user wants to create a new wake word from scratch with wakeword-lab (e.g. "create a wake word", "train a wake word for X", "start a new wake-word project"). Covers project init, synthetic data bootstrap (TTS positives, confusable negatives, speech negatives, real noise), and the first baseline training run.
---

# Wake-word project setup

Bootstrap a new wake-word project. All commands run from the repository root with the installed CLI (`wakeword-lab`, or `./venv/bin/wakeword-lab` in a dev checkout). `<proj>` is the new project directory.

## Steps in order

```bash
wakeword-lab init <proj> --wake-word 'Hey Nova'
wakeword-lab synthesize <proj> --phrase 'Hey Nova'                      # TTS positives (network)
wakeword-lab synthesize <proj> --phrase 'Hey Nova' --locales it-IT      # optional accented pool
wakeword-lab synthesize-negatives <proj> --phrase 'Hey Box' --phrase 'Hey Fox'   # confusables (network)
wakeword-lab fetch-negatives <proj> --count 200                         # real speech negatives (network)
wakeword-lab fetch-noise <proj>                                         # DEMAND ambience (network)
```

Then the user MUST record real audio before the first real training — TTS-only models do not generalize to human voices (validation accuracy stalls ~35%). Point them at the `wakeword-data-collection` skill.

With at least some real takes present:

```bash
wakeword-lab prepare <proj>
wakeword-lab experiment <proj> --run-name baseline
```

## Rules that prevent the most common failures

- **TTS audio is training-only.** Never use synthesized clips for validation/test; the pipeline marks them train-only on purpose.
- **`prepare` fails unless every split has positive AND negative examples.** A fresh project has everything pinned/auto-assigned to train; real recordings pinned to validation/test are required. This is expected, not a bug.
- **Choose confusable negative phrases that differ by one phoneme** from the wake word ("Hey Box"/"Hey Fox" for "Hey Vox"), plus generic commands ("Play music", "Stop the timer"). 8–12 phrases at low variants-per-locale is plenty.
- **Accents matter.** If the user speaks with a non-US accent, add `--locales` pools (e.g. `it-IT`, `en-IN`) for matching-accent TTS.
- Network steps are only the synthesize/fetch commands; everything else is local. Edge TTS occasionally drops an utterance — the CLI retries then skips, that's fine.

## Verify setup before training

- `data/raw/positive/` non-empty; `data/raw/negative/` non-empty; `data/raw/noise/` has files
- `data/raw/metadata.csv` exists; TTS rows all say `train`
- User has recorded (or committed to recording) real validation/test takes
