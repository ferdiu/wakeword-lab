# Wakeword Lab

A local-first research repository for experimenting with small custom wake-word classifiers and preparing models for Android. It implements the reproducible core (audio ingestion, log-mel features, source-group splitting, DS-CNN training, validation thresholding, TFLite conversion, metrics) and includes an Android PCM ring-buffer starter. It is **not yet a drop-in production Android detector**: the Android LiteRT inference adapter, full streaming `AudioRecord` controller, feature parity implementation and end-to-end device benchmark remain follow-up work. Do not mistake the starter module for a complete AAR SDK.

The motivating Medium post is [Building a Custom Wake Word Detector for Android — Without a Cloud or API Key](https://medium.com/@patare.vivek/building-a-custom-wake-word-detector-for-android-without-a-cloud-or-api-key-7983e752362b). Its concrete ideas and the parameters it leaves unspecified are mapped in [docs/ARTICLE_MAPPING.md](docs/ARTICLE_MAPPING.md).

## Quick start

Python 3.10+; install the core package and tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
wakeword-lab init ./my-keyword --wake-word 'Hey Nova'
```

Add WAV recordings to `data/raw/positive/` and `data/raw/negative/`. Prefer many independent speakers, devices, rooms and distances. Name files with a stable source prefix such as `speaker01__take01.wav`, or provide `data/raw/metadata.csv` with `path,speaker_id`. The same speaker/source must share one ID across positive and negative recordings. Use a genuinely separate source for test data.

```bash
wakeword-lab prepare ./my-keyword
pip install -e '.[train]'
wakeword-lab train ./my-keyword
wakeword-lab export ./my-keyword
wakeword-lab evaluate ./my-keyword
wakeword-lab report ./my-keyword
```

`prepare` creates one-second windows (20% hop for long recordings), extracts 98x40 log-mel tensors and writes group-disjoint train/validation/test sets. Training computes per-mel normalization from train only. Export creates float and full-integer INT8 TFLite files; calibration uses up to 200 training windows. Threshold selection uses validation only. Test evaluation never tunes the threshold.

## Important limitations and methodology

- The article reports 16 kHz, 10 ms capture chunks, a one-second rolling window, 20 ms scoring, log-mel, DS-CNN, energy VAD, Edge TTS, augmentation and label smoothing. It does **not** specify the mel filter convention, frame length, FFT size, exact DS-CNN topology, split protocol or full source. This repository makes defaults explicit; they are engineering choices, not recovered article settings.
- The current preparation pipeline does not yet implement the article's full augmentation suite or TTS conversion pipeline. `src/wakeword_lab/synthesize.py` is an optional Edge TTS voice generator; its MP3 output must be converted to 16 kHz mono WAV before `prepare`.
- `data/raw/noise/` is reserved but not yet mixed into training. Real ambient negatives should be added as negative recordings. Synthetic noise is not a substitute.
- The Android module currently provides tested PCM buffering/configuration primitives only. It does not claim to perform microphone capture, VAD, frontend extraction, LiteRT inference or lifecycle management.
- The article's 42 KB, <5 ms and 100% test accuracy are author-reported results, not guarantees. Report false accepts/hour on long continuous held-out recordings and benchmark on target phones.
- `choose_threshold` is a simple window-level FPR constraint. Event-level false activations/hour requires a continuous-stream evaluator and is not inferred from window accuracy.

## Tests

```bash
pip install -e '.[dev]'
pytest
```

Training/export tests require `.[train]`. Android JUnit tests require JDK 17, Gradle 8.7+ and Android SDK 35; run `gradle -p android :wakeword-sdk:testDebugUnitTest`.

## Privacy and licensing

The Python pipeline processes local files; Edge TTS synthesis contacts Microsoft's service and is optional. Obtain consent for voice recordings and check dataset/TTS terms. Do not commit private recordings, generated audio or speaker-identifying metadata. Apache-2.0.
