# Android SDK and demo

This project contains `:wakeword-sdk` (an Android library) and `:demo` (a minimal user-triggered sample app). It uses Google's standalone LiteRT 2.3.2 `CompiledModel` API on CPU. The SDK supports minSdk 24 and compileSdk 36.

## Build

Prerequisites: JDK 17, Gradle 8.13, Android Gradle Plugin 8.13.2, Kotlin Gradle Plugin 2.4.21 and Android SDK Platform 36 / Build Tools 36.0.0.

```bash
# From repository root, after training/exporting a model:
wakeword-lab package-android ./my-wake-word --run-name baseline
gradle -p android :wakeword-sdk:testDebugUnitTest :wakeword-sdk:assembleRelease :demo:assembleDebug
```

The Gradle build creates the AAR and demo APK. The model is not committed: package `wakeword_int8.tflite` and its matching `model_metadata.json` into the consuming application's assets. The CLI above copies both into the demo assets folder.

## Runtime design

- `AudioRecord` captures mono PCM16 at 16 kHz in 10 ms blocks on a dedicated thread set to `THREAD_PRIORITY_URGENT_AUDIO`.
- `AudioRingBuffer` provides a rolling one-second context.
- Energy RMS is checked every 20 ms. Quiet windows do not run inference.
- A dedicated, single-thread executor owns LiteRT `CompiledModel`, tensor buffers, warm-up, inference and close. There is at most one inference task pending, so slow models do not build an unbounded queue behind real-time capture.
- The rolling log-mel frontend reuses 96 of 98 frames on contiguous 20 ms steps and falls back to full extraction after audio gaps.
- INT8 input quantization uses the exact scale/zero point from model metadata; output is dequantized using the exported output scale/zero point before thresholding.
- Cooldown prevents repeated callbacks; callbacks and errors are delivered on the main looper.
- `WakeWordListeningService` is an optional foreground service with an ongoing notification and Stop action. Start it from a visible Activity after `RECORD_AUDIO` permission has been granted.

Android 14+ requires declaring the microphone foreground-service type and `FOREGROUND_SERVICE_MICROPHONE` permission. Microphone FGS creation is subject to while-in-use restrictions; do not start it silently from background or boot. This SDK does not bypass Android's privacy/permission model.

## Public entry points

- `WakeWordDetector(context, listener, config)` for applications managing their own foreground-service lifecycle.
- `WakeWordListeningService` for the ready-made service integration.
- `WakeWordModelBenchmark.run(context, pcmWindows)` for background-thread latency/model-size measurements.

The Python and Kotlin log-mel frontends are locked by `wakeword-sdk/src/test/resources/frontend_golden.csv` and the Kotlin `LogMelFrontendTest`. Regenerate the fixture with `python scripts/generate_frontend_fixture.py` only after an intentional frontend change, then run both Python and Android tests.

There is no physical-device CI. Test on target devices for load time, p50/p90/p99 latency, microphone routing, CPU, thermals, audio interruptions and long-duration battery drain.


## Runtime selection

This starter bundles the LiteRT runtime instead of using the Google Play services runtime. The reason is portability: the SDK is intended to remain usable on Android devices without Google Play services (including some AOSP and region-specific builds). Inference is still entirely on-device and makes no network requests. If your app targets only Google-certified devices and minimizing APK size is more important than GMS independence, consider switching to `com.google.android.gms:play-services-litert`; do not package both runtimes in one build variant. The SDK initializes a context-aware `Environment` once per process (enabling the runtime compiler cache) and shares it across model sessions and confines each model's creation, inference, and cleanup to its own serial executor.
