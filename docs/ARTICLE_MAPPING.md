# Mapping the Medium article to this lab

The article (May 18, 2026) describes:

| Article idea | Article detail | Repository status |
|---|---|---|
| Streaming capture | 16 kHz mono, 10 ms chunks, rolling 1 s, score every 20 ms | Android config/ring-buffer primitives only; capture loop pending |
| Frontend | log-mel spectrogram | Python frontend implemented; 25 ms frame, 10 ms hop, 512 FFT, 40 bands are explicit lab choices because the post omits them |
| Classifier | DS-CNN, confidence threshold 0.80 | Compact DS-CNN training implementation; threshold calibrated on validation instead of fixed 0.80 |
| VAD | chunk energy gate | Not yet implemented in Android |
| Synthetic positives | Edge TTS; en-IN voices weighted 3x | Optional voice-list/synthesis prototype; MP3-to-WAV conversion and weighted sampling still need hardening |
| Augmentation | 10 copies; speed 0.8–1.2, pitch ±4 semitones, shift ±200 ms, reverb, noise | Not yet implemented; explicitly not claimed |
| Label smoothing | target confidence 0.95 | Keras BCE label_smoothing=0.05 |
| Quantization | 200 representative samples (100 positive/100 negative) | Full INT8 conversion with up to 200 training windows; current sampler is not yet class-balanced |
| Android threading | urgent audio capture thread, inference worker, main-thread callback | Not yet implemented |
| Reported result | 42 KB, <5 ms, 100% test accuracy | Not assumed; must be measured on independent test audio and target devices |

## Critical experimental recommendations

1. Split by speaker/source before windowing or augmentation. Never put augmented copies of a source in a different split.
2. Keep validation for threshold/model selection and test as a one-time final estimate.
3. Collect long, continuous negative audio from realistic conditions. Random-window accuracy is not an adequate wake-word metric.
4. Include quiet rooms, fans, TV, music, nearby speech, far-field speech, clipping, reverberation and microphone-specific recordings.
5. Compare float and INT8 scores on the same test set; inspect false activations/hour and missed wake-word rate.
6. Measure cold-start, steady-state latency, CPU load, memory and battery on each target device.
