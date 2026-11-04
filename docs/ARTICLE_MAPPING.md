# Article-to-repository mapping

The reference article is [Building a Custom Wake Word Detector for Android — Without a Cloud or API Key](https://medium.com/@patare.vivek/building-a-custom-wake-word-detector-for-android-without-a-cloud-or-api-key-7983e752362b). It is a high-level build narrative. It does not publish a complete repository, training corpus, exact DS-CNN layer specification, log-mel coefficient recipe, data split protocol, or full benchmark harness. The implementation below distinguishes article statements from choices this repository makes where the article is underspecified.

| Article concept | What the article says | Wakeword Lab implementation |
|---|---|---|
| Audio capture | 16 kHz mono, 10 ms chunks | `AudioRecord` reads 160-sample PCM16 chunks at 16 kHz; converted to float [-1,1] |
| Rolling audio | One-second continuously updated context | 16,000-sample synchronized circular buffer, chronological snapshot and left padding |
| Inference cadence | Check every 20 ms | 320-sample score cadence (two 10 ms capture chunks); at most one inference task is pending |
| Features | Log-mel spectrogram | Python/Kotlin matched frontend: 25 ms Hann frame, 10 ms hop, 512 FFT, 40 HTK mel bands, 60–7,600 Hz, natural log mel power; these numerical frontend settings are explicit project choices because the post does not specify them |
| Classifier | DS-CNN, score between 0 and 1 | Compact depthwise-separable CNN with global average pooling and sigmoid; width/dropout are configurable |
| Threshold | Example threshold 0.80 | Calibrated from predictions made by the exported INT8 model on validation data; test data is not used to set it |
| VAD | Skip model when chunk energy is low | RMS gate with configurable threshold; stream evaluator replays the same 20 ms cadence and gate |
| Positive corpus | Microsoft Edge TTS; many accents/voices; en-IN 3× weighted | Optional TTS command chooses locale using 3:1 en-IN weighting over en-US/en-GB/en-AU, randomizes rate/pitch, converts result to 16 kHz mono WAV and records speaker/locale metadata and a coarse energy-VAD phrase interval used to crop the complete TTS phrase into the one-second context |
| Positive augmentation | 10 copies; speed 0.8–1.2×; pitch ±4 semitones; shift ±200 ms; room reverb and background noise | Training-only augmentation implements those transformation ranges; librosa performs duration-preserving pitch shifts when installed (the training extra); FFT reverb and SNR-scaled real noise mixing use train-partition noise only |
| Negative corpus | Silence and ambient noise; 400 synthetic background-noise clips mentioned as a fix | `generate-noise` creates seeded colored-noise/fan/hum WAVs; noise snippets become train-only negative examples and augmentation backgrounds. Negative WAVs provide independent held-out hard negatives from real environments |
| Label smoothing | Soft targets with positive ceiling around 0.95 | Config epsilon 0.05 maps to binary targets 0.05 / 0.95 (Keras BCE argument 0.10) |
| Quantization | 200 calibration examples, 100 positive + 100 negative | Balanced representative data sampled from training only; full-integer INT8 input/output and weights; actual scales/zero-points written to metadata |
| INT8 evaluation | Article says no observed test accuracy loss in its dataset | Export compares float/INT8 validation scores; test evaluation uses quantized model and validation-calibrated threshold |
| Android threads | Urgent-priority capture, separate inference worker, main-thread callbacks | Dedicated capture executor and serial model executor; one outstanding inference task; callbacks/errors posted through main looper |
| Cooldown | SDK includes cooldown management | Configurable cooldown in milliseconds prevents repeated callbacks for one phrase |
| Reported model size/latency | 42 KB and under 5 ms reported by article | Model bytes and p50/p90/p99 are measured in artifacts/device benchmark; no hardware-independent performance claim |

## Article's reported pitfalls and responses

1. **Silent-room false positives:** the article attributes these to missing ambient examples and per-clip normalization. The lab computes global per-mel normalization from training features only, includes hard negative recordings, permits background mixing, and reports continuous false activations/hour. It does not apply per-recording or per-window amplitude normalization.
2. **TTS-to-human gap:** the article reports weaker scores on real voice than on generated voice and recommends real enrollment samples. This repository supports speaker-group splits and TTS pinned to train; add real recordings from the intended user to training and independent speakers/rooms to dev/test. Do not report TTS-only test scores as real-world performance.
3. **Keras 2/Keras 3 compatibility:** the training extra targets modern TensorFlow/Keras (TensorFlow >=2.16) and the export smoke test is part of CI.
4. **INT8 accuracy:** balanced representative calibration is applied only to training features; validation is used for threshold and quantization drift; final test remains untouched until `evaluate`.

## Engineering decisions not determined by the article

- Mel scale, mel-band count, frequency range, frame length, FFT size and exact window convention.
- Exact DS-CNN layer widths, stride schedule, dropout and learning rate.
- Speaker/source grouping, phrase-boundary timestamps for positive recordings, and how validation/test partitions are formed.
- Whether to use float, dynamic-range or full-integer quantization; this lab uses full-integer INT8 for CPU deployment.
- How to estimate operational false triggers per hour; this lab replays continuous holdout audio with the Android-style VAD/cooldown.
- Android API/runtime and microphone foreground-service details; this lab uses LiteRT `CompiledModel` and Android's microphone foreground-service type with explicit user initiation.

The measured results depend on data and target device. The article's 42 KB, under-5 ms and 100% accuracy are not guarantees.

## Label smoothing detail

The config `label_smoothing: 0.05` expresses the desired binary targets directly: negative 0.05 and positive 0.95, as described by the article. Keras `BinaryCrossentropy` interpolates a target with 0.5, so the implementation passes `label_smoothing=0.10` to Keras to produce those targets. This avoids the subtle 0.025/0.975 targets produced by passing 0.05 straight to Keras.
