# Wakeword Lab

**A local-first laboratory for training, evaluating, quantizing and deploying very small custom wake-word detectors on Android.** The repository includes the Python preparation/training/export/evaluation workflow and an Android SDK/demo implementation using LiteRT's `CompiledModel` API. It does not include a pretrained model or voice dataset: bring recordings you have permission to use, train a model, and package the result. The Python unit suite and frontend parity are verified locally; the full TensorFlow training/export and Android Gradle builds are wired into CI but require dependencies and SDK components not installed in this environment.

The project is based on the ideas in [Building a Custom Wake Word Detector for Android — Without a Cloud or API Key](https://medium.com/@patare.vivek/building-a-custom-wake-word-detector-for-android-without-a-cloud-or-api-key-7983e752362b). The article is a high-level description, not a reproducible research artifact. See [the article mapping](docs/ARTICLE_MAPPING.md) for what is directly described versus what this lab necessarily specifies itself.

## What is included

- WAV ingestion, mono conversion and 16 kHz resampling.
- A deterministic 98 × 40 log-mel frontend with a Python/Kotlin golden parity test.
- Source/speaker-group train/validation/test splits; optional split pins in metadata.
- Training-only speed, pitch, time-shift, room-reverb and real-noise augmentation.
- Optional Microsoft Edge TTS generation with en-IN locale oversampling and WAV conversion.
- A compact DS-CNN, train-only per-mel normalization, class weighting and early stopping. The configured binary smoothing epsilon (0.05) maps explicitly to 0.05/0.95 targets, matching the article's stated 95% positive target.
- Float32 and full-integer INT8 LiteRT/TFLite export, balanced representative calibration, validation-only threshold calibration, and float/INT8 score-drift reporting.
- Test-set metrics and a continuous-stream false-activation/hour evaluator.
- Reproducible experiment folders and width/dropout sweeps.
- Android microphone capture, energy VAD, one-second circular PCM window, 20 ms inference cadence, rolling log-mel reuse (only two new frames per contiguous step), LiteRT CPU inference, cooldown, main-thread callbacks, foreground-service integration and benchmark helper.
- Python and Android CI workflows. Device latency, power and acoustic quality still have to be measured on target phones.

## 1. Install and create a project

Python 3.10+ is required. The lightweight dependency set is separate from TensorFlow training and optional TTS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev,train]'
wakeword-lab init ./my-wake-word --wake-word 'Hey Nova'
```

Generate a reproducible starter set of 400 low-level synthetic background clips (colored noise plus fan/hum profiles) using only the core dependencies:

```bash
wakeword-lab generate-noise ./my-wake-word --count 400 --seed 42
```

These are useful extra training negatives and augmentation sources, but they are not a substitute for real room, television, music and device recordings. Keep genuine ambient holdouts under `negative/` for validation/test.

For optional Edge TTS synthesis, install the extra and the `ffmpeg` executable:

```bash
pip install -e '.[tts]'
# Install ffmpeg with your OS package manager, then:
wakeword-lab synthesize ./my-wake-word --phrase 'Hey Nova' --variants-per-locale 24
```

Edge TTS makes network requests to Microsoft's service. The rest of the pipeline runs locally. TTS utterances are marked as **training-only** in `metadata.csv`; a coarse energy-VAD estimates phrase boundaries so the complete utterance is kept inside the one-second training window. Phrases that cannot fit are rejected. Inspect the generated WAVs and timing metadata; this is not a general forced aligner. Do not use synthetic variants as held-out test data.

For optional interactive microphone recording, install the record extra (sounddevice + soundfile):

```bash
pip install -e '.[record]'
```

## 2. Prepare audio data

Put positive recordings of the wake phrase in `data/raw/positive/`, non-wake speech and ambient audio in `data/raw/negative/`, and optional background-only recordings in `data/raw/noise/`. Input WAV files may be mono/stereo and at supported sample rates; they are resampled/downmixed on ingest. Use real recordings from different speakers, distances, devices and rooms. Keep the final test set independent from choices made during training.

`metadata.csv` is optional but strongly recommended. It uses one row per recording:

```csv
path,speaker_id,group_id,split,locale,phrase_start_s,phrase_end_s
my_take_01.wav,my_voice,my_voice_session_01,train,en-GB,0.35,0.92
my_take_heldout.wav,my_voice,my_voice_session_02,test,en-GB,,
other_speaker.wav,speaker_heldout_01,speaker_heldout_01,test,en-GB,,
```

`phrase_start_s` and `phrase_end_s` are optional timestamps (seconds) for the complete wake phrase in a positive recording. When present, preprocessing samples several one-second contexts that all contain the entire phrase; phrases longer than one second are rejected because they cannot fit the current model input. Without timestamps, a positive recording longer than one second is treated as one isolated centered utterance and center-cropped once. Do not put long multi-phrase speech recordings in `positive/`: label the actual phrase interval or trim the WAV to one centered utterance. This avoids silently labelling unrelated or partial speech as positive.

`path` is matched by filename. `speaker_id` records who spoke. By default, it is also the split group, so all recordings from one speaker stay in one partition. Optional `group_id` overrides that split unit: use it for a recording session when evaluating a personalized detector on new sessions from the same speaker (as in the example), or set it equal to `speaker_id` for strict speaker-independent evaluation. Keep positive and negative recordings from the same capture session in the same `group_id`. `split` may be blank or one of `train`, `validation`, `test`; it pins the entire group to that split. The default when metadata is missing is the part of the filename before `__` (for example, `speaker01__take02.wav`). Do not accidentally give clips from the same recording session different group IDs.

The `noise/` folder contains background-only recordings. By default, short snippets from training-partition noise recordings are added as pure negative training examples and are also eligible to be mixed into augmented positives. `wakeword-lab generate-noise` creates synthetic colored-noise/fan/hum examples here. They never enter validation/test by default. Put independent held-out ambient recordings in `negative/` when you want them to contribute to validation/test false-accept estimates.

### Recording from a microphone

With the record extra installed, capture takes interactively straight into the raw folders:

```bash
wakeword-lab record ./my-wake-word --kind positive --speaker alice --count 20
```

Each take is saved as a 16 kHz mono pcm16 WAV named `<speaker>__rec_<runid>_<index>.wav` and tracked in `metadata.csv` (one session group per run). Takes are capped at 10 seconds; takes shorter than 0.3 s are rejected and can be redone without consuming the count. `--kind` also accepts `negative` and `noise`. Use `--split train|validation|test` to pin the whole session group to one partition; leave it blank to let the automatic group splitting assign it. When `--count` is omitted you are prompted (default 20).

### Recording with the phone app

For training data captured in real on-device conditions (the same microphone and audio path as the detector), build the standalone recorder app and capture takes straight on the phone:

```bash
wakeword-lab build-recorder ./my-wake-word          # add --install with a device attached
wakeword-lab fetch-recordings ./my-wake-word --speaker alice --delete
```

`build-recorder` compiles `android/recorder/` with the project wake word baked into the UI and prints the APK path; `--install` also installs it on the attached device. The app shows six push-to-talk buttons (positive/negative × train/validation/test): hold a button to record (min 0.3 s, max 10 s), release to save, slide off to cancel. `fetch-recordings` pulls the recorded WAVs over adb into `data/raw/<kind>/` and pins each take's split from its on-device folder, grouped per pull as `<speaker>_phone_<batch>`; `--delete` removes the on-device copies after a successful pull. Because these takes come from a real phone microphone, they match on-device deployment conditions better than studio recordings.

### Synthetic hard negatives

Generate non-wake speech utterances with the same Edge TTS voices as positive synthesis:

```bash
wakeword-lab synthesize-negatives ./my-wake-word \
  --phrase 'hello there' --phrase 'what time is it' --variants-per-locale 8
# or: --phrases-file negatives.txt  (one phrase per line, # comments allowed)
```

Each phrase yields `variants_per_locale * 6` utterances and clips are written to `negative/` with a `neg_tts_` prefix, marked **training-only** in `metadata.csv` without phrase timestamps — negatives are windowed at the configured hop during preparation. Requires the `tts` extra and `ffmpeg` like positive synthesis.

### Dataset hard negatives

Fetch one-second command-word clips from TensorFlow's mini_speech_commands archive as hard negatives:

```bash
wakeword-lab fetch-negatives ./my-wake-word --count 200 --seed 42
```

This downloads the ~174 MB archive (one of the network steps besides TTS and `fetch-noise`, with a 300 s timeout), selects clips uniformly across the eight command-word folders and copies them into `negative/` as `sc_<word>__<original>.wav`. The filename-prefix grouping then keeps every command word inside one split partition. No metadata rows are added for these clips; the download is cached nowhere — rerun with the same seed for a reproducible selection.

### Real-world noise (DEMAND)

Fetch genuine ambience from the DEMAND database (Zenodo record 1227121) into `noise/`:

```bash
wakeword-lab fetch-noise ./my-wake-word
# optional: --environments DKITCHEN,OOFFICE,STRAFFIC
```

Each 16 kHz mono environment archive (~5 minutes of continuous room noise, 200–300 MB) is SHA-256 verified and its WAVs are extracted as `demand_<env>__<original>.wav`. Every environment gets a metadata row with its own `speaker_id`/`group_id` and a split pin assigned round-robin (`train`, `validation`, `test`, …) so each split owns at least one noise group; edit the `split` column in `metadata.csv` to reroute an environment. Environments without a pinned hash fail closed.

With the `noise_mixing` section of `wakeword.yaml` enabled, every windowed clip in **every** split (train, validation and test) keeps its original plus `copies_per_clip` extra copies mixed with noise drawn only from that same split's own noise groups, at a random SNR in `[snr_min_db, snr_max_db]`:

```yaml
noise_mixing:
  enabled: false
  copies_per_clip: 1
  snr_min_db: 0.0
  snr_max_db: 20.0
```

This is disabled by default; when enabled, a partition without any noise group is skipped with a warning. Speech-domain augmentation (speed/pitch/shift/reverb) stays train-only; noise mixing is the only validation/test augmentation and never mixes noise across split boundaries.

```bash
wakeword-lab prepare ./my-wake-word
```

The command first splits source groups and then windows/features each partition. It produces `data/processed/{train,validation,test}.npz` and `split_manifest.json`. Positive recordings are cropped so the complete wake phrase fits the one-second context (using phrase timestamps where supplied); negative recordings are sampled at the configured hop, with a per-recording cap to keep preparation bounded. The augmentation count and window cap are configurable in `wakeword.yaml`. Speech-domain augmentation (speed/pitch/shift/reverb) applies to the training partition only; when `noise_mixing` is enabled, validation and test windows additionally receive noise-mixed copies built exclusively from their own split's pinned noise groups.

**Data quality requirement:** each of train, validation and test must contain positive and negative examples. If not, the command fails rather than generating a misleading model. Add more independent groups or explicitly pin suitable recordings. For robust speaker-generalization estimates, aim for many speakers in all three splits; three groups is only the mechanical minimum, not a statistically good dataset.

## 3. Train and run experiments

Run one experiment end to end:

```bash
wakeword-lab experiment ./my-wake-word --run-name baseline
```

This runs preparation when processed data is absent, trains the DS-CNN, exports float32 and INT8 models, calibrates a threshold from validation predictions made by the exported INT8 model, and evaluates that model once on the untouched test partition. Each run is isolated under `artifacts/experiments/<run-name>/`.

For lower-level control:

```bash
wakeword-lab prepare ./my-wake-word --no-augmentation
wakeword-lab train ./my-wake-word --run-name width16 --width 16 --dropout 0.10
wakeword-lab export ./my-wake-word --run-name width16
wakeword-lab evaluate ./my-wake-word --run-name width16
wakeword-lab report ./my-wake-word
wakeword-lab compare ./my-wake-word
```

Try a small architecture sweep:

```bash
wakeword-lab sweep ./my-wake-word --widths 8,12,16,24 --dropouts 0.1,0.15,0.2
wakeword-lab compare ./my-wake-word
```

Sweeps rank candidates using **validation** metrics and model size only; they do not evaluate the final test set. After choosing a candidate, run `wakeword-lab evaluate ./my-wake-word --run-name <chosen-run>` once for the final held-out test result. The convenience `experiment` command is for a single run where you explicitly want training, export and final test evaluation in one go.

Edit `wakeword.yaml` to change augmentation, split fractions, frontend contract, model training, calibration count and target validation FPR. The Android frontend contract is deliberately fixed at 16 kHz / 1 s / 98 × 40; configuration validation rejects incompatible frontend settings. If you change split or augmentation settings, rerun `prepare` before training.

Each run folder includes the training CSV, best and final Keras model, normalization metadata, run config, float32 and INT8 `.tflite` files, model metadata, validation metrics, test metrics and summary.

Every experiment also generates a `report/` subdirectory (and `wakeword-lab report <project> --run-name X` refreshes one on demand; plain `report` prints summaries and refreshes all runs that have one). The report contains four plots and a markdown summary:

- `training_curves.png` — train vs validation loss and metrics per epoch, with the early-stopping end marked
- `score_distributions.png` — overlaid positive/negative INT8 score histograms for validation and test, with the calibrated threshold
- `roc_pr.png` — validation ROC (FPR vs recall) and precision/recall vs threshold, with the calibrated threshold point marked
- `quantization_drift.png` — paired float32 vs INT8 scores on validation with MAE/max-abs-error annotations
- `report.md` — run config, seed, data-manifest hash, epochs completed, model size, quantization MAE and per-split metrics at the calibrated threshold

Plotting needs matplotlib, which ships with the `train` extra; a failed report never fails the experiment itself. INT8 calibration samples are class-balanced (100 positive + 100 negative by default), drawn only from training features. The validation threshold is selected to meet the configured empirical window FPR, not hard-coded to 0.80. When validation has too few negatives to resolve the requested FPR, export warns; it does not pretend that the target has been proven.

## 4. Evaluate continuous audio (important)

Window-level accuracy does not tell you how often an always-listening detector will wake the app in a quiet room or during television/music. Evaluate hours of independent, continuous recordings:

```bash
wakeword-lab evaluate-stream ./my-wake-word ./held-out-recordings --run-name baseline --vad-rms 0.008
```

Optionally annotate wake-word intervals in `annotations.json`:

```json
{
  "living_room.wav": [
    {"start": 15.2, "end": 16.8},
    {"start": 91.0, "end": 92.3}
  ],
  "office_negative.wav": []
}
```

```bash
wakeword-lab evaluate-stream ./my-wake-word ./held-out-recordings \
  --run-name baseline --annotations ./held-out-recordings/annotations.json
```

The replay expects 16 kHz WAV recordings (mono or multichannel; channels are downmixed), matching Android capture. Convert other sample rates before evaluation, for example `ffmpeg -i room.wav -ar 16000 -ac 1 room_16k.wav`. The replay uses the same one-second context, 20 ms score cadence, energy gate and cooldown as the Android SDK. The report includes false activations/hour, missed/ detected annotated phrases and per-recording events. Only recordings of at least one second are evaluated. Use several hours across different rooms/devices; a few minutes cannot establish a low false-activation rate. Tune threshold/VAD/cooldown on development recordings only, not on the final holdout.

## 5. Package the Android demo / SDK

A trained model is intentionally not committed. Package a chosen experiment into the demo app's assets:

```bash
wakeword-lab package-android ./my-wake-word --run-name baseline
# or: python scripts/package_android.py ./my-wake-word --run-name baseline
```

Build with JDK 17, Gradle 8.13, Android Gradle Plugin 8.13.2, Kotlin Gradle Plugin 2.4.21 and Android SDK Platform 36:

```bash
gradle -p android :wakeword-sdk:testDebugUnitTest :wakeword-sdk:assembleRelease :demo:assembleDebug
```

Outputs:

- `android/wakeword-sdk/build/outputs/aar/wakeword-sdk-release.aar`
- `android/demo/build/outputs/apk/debug/demo-debug.apk`

The demo provides explicit Start/Stop controls. The microphone is not started until the user taps Start and grants `RECORD_AUDIO`. The library declares a microphone foreground service for an ongoing user-visible session. On Android 14+, microphone foreground-service creation is subject to while-in-use permission restrictions: start it from a visible Activity after permission, not from boot or a silent background job. A persistent notification includes a Stop action.

The bundled standalone LiteRT runtime uses the current `CompiledModel` API. It is bundled rather than delegated to Google Play services so the SDK can also run on Android builds without GMS; this is a portability trade-off that increases APK size. Inference runs on CPU by default to prioritize compatibility and predictable INT8 support; the SDK can be adapted to select another accelerator after benchmarking. One process-lifetime LiteRT `Environment` is shared, while `CompiledModel`, input/output buffers and inference are confined to a single serial executor. Capture runs at urgent-audio priority and never waits for model initialization or inference. At most one inference can be queued at a time. Detection/error callbacks are posted to the main thread.

### Integrate into your own app

Add the SDK AAR (or use the Gradle project dependency), ensure the two files `wakeword_int8.tflite` and `model_metadata.json` are in your app's `src/main/assets`, and declare the microphone permissions/service through the SDK manifest merger. If you copy a raw AAR directly rather than consuming the Gradle module or a Maven-published artifact with its POM, also add `implementation("com.google.ai.edge.litert:litert:2.3.2")` to the consuming app: a raw AAR does not embed its transitive Maven dependencies. Ask the user for runtime `RECORD_AUDIO` permission, then start the listening service from a visible Activity:

```kotlin
val intent = Intent(this, WakeWordListeningService::class.java)
if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(intent)
else startService(intent)
```

Or construct `WakeWordDetector(context, listener)` and host it in your own appropriately declared microphone foreground service. Call `start()` only after permission; call `stop()` to end capture and `close()` when the detector will no longer be used. The app must keep model and metadata from the same export run together; metadata contains exact normalization and affine quantization parameters.

### Device benchmark

`WakeWordModelBenchmark.run(context, pcmWindows)` reports model load/warm-up time, frontend p50, model inference p50, end-to-end p50/p90/p99 and packaged model size. Call it from a background thread in a release build with representative one-second recordings after an initial warm-up. To measure the rolling-feature fast path, pass chronologically adjacent windows shifted by 20 ms; non-contiguous windows intentionally trigger full frontend recomputation. Battery drain needs a longer device test with Android Studio Profiler / `dumpsys batterystats`; inference latency alone is not a battery-life measurement.

## 6. Frontend contract

Python and Kotlin are locked together by `android/wakeword-sdk/src/test/resources/frontend_golden.csv`, generated by `scripts/generate_frontend_fixture.py`. The test's deterministic 1-second two-tone signal matches with a maximum absolute feature error below `1e-5` in the local Kotlin standalone check.

| Parameter | Value |
|---|---:|
| PCM | 16 kHz mono float, range [-1, 1] |
| Context | 16,000 samples / 1 second |
| Capture chunk | 160 samples / 10 ms |
| Score cadence | 320 samples / 20 ms |
| Window | 400 samples / 25 ms symmetric Hann |
| FFT | 512-point real FFT |
| Mel scale | HTK, 40 bands, 60–7,600 Hz |
| Feature | Natural log of mel power, floor `1e-6` |
| Tensor | `[1, 98, 40, 1]` |

For contiguous 20 ms windows, Android reuses the 96 overlapping log-mel frames and computes only two new frames; after a VAD gap or non-contiguous audio step, it falls back to full feature extraction. A rolling-window test compares this fast path against the full frontend. The same model file and `model_metadata.json` must be used together. Android reads the actual INT8 input scale/zero point and output scale/zero point from that metadata and does not assume symmetric quantization. Do not change either frontend independently; update and run the golden test on both platforms.

## 7. Tests and CI

```bash
pip install -e '.[dev]'
timeout 90s pytest -q
```

The local unit suite covers resampling, ring/window behavior, frontend output, deterministic augmentation, train-only augmentation, group isolation, split pins, metrics, FPR threshold selection, configuration validation and stream interval validation. `tests/test_tf_integration.py` is skipped unless TensorFlow is installed; the `training-export-smoke` CI job installs the training extra and exercises one-epoch training, INT8 export and test evaluation with a hard timeout. Android CI builds the SDK/demo and runs Kotlin/JUnit ring-buffer and frontend golden tests with a bounded job timeout.

## Methodology and limitations

The article does not publish its dataset, exact DS-CNN source/topology, frontend coefficients, held-out split protocol, or complete benchmark procedure. This project has explicit defaults rather than pretending those missing details can be reconstructed. The article's 42 KB, under-5 ms and 100% test-accuracy figures are results claimed for its author's particular corpus and device, not an acceptance target or guarantee here. A real wake-word model needs a range of human voices, hard negatives, continuous ambient data and an independently held-out test set.

## Privacy and licensing

Audio processing, training, export and Android inference are local. Optional Edge TTS synthesis uses Microsoft's network service. Obtain consent for voice recordings and check the license/terms for every imported dataset and TTS service. Do not commit private recordings, generated voice files, speaker-identifying metadata, trained models or evaluation audio. Apache-2.0; see [LICENSE](LICENSE).
