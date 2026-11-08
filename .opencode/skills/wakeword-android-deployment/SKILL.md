---
name: wakeword-android-deployment
description: Use when deploying a trained wake-word model to Android (package-android, gradle build/install, demo app), when on-device behavior differs from desktop metrics (misses, false fires, scores shifted), when tuning the detection threshold on device, or when debugging the demo app via logcat heartbeat/detection logs. Covers the JAVA_HOME 17 build trap.
---

# Wake-word Android deployment

## Package and install

```bash
wakeword-lab package-android <proj> --run-name <name>   # copies wakeword_int8.tflite + model_metadata.json into android/demo assets

export JAVA_HOME=~/.gradle/jdks/eclipse_adoptium-17-amd64-linux.2   # REQUIRED: system Java is newer, AGP rejects it
cd android && ./gradlew installDebug --no-daemon                    # needs a device visible in `adb devices`
```

If `~/.gradle/jdks/...` does not exist, any JDK 17 works — set JAVA_HOME to it. CI builds with JDK 17 + Gradle 8.13; the repo has a wrapper (`android/gradlew`).

## The demo app (dev.wakeword.demo)

- Start/Stop buttons, live log view, threshold slider (0.20–0.90, initialized from the packaged metadata).
- The notification must be visible while listening; if it's missing, the notification runtime permission (Android 13+) was denied — tap Start again to re-prompt.
- The log shows the same lines as logcat tag `WakeWordDetector`:
  - `heartbeat: maxScore=0.559 threshold=0.52` — every ~2 s of non-silent audio; the max model score in that window
  - `wake word detected, score=0.87`
  - `detector error: ...`

## Reading on-device behavior

Desktop-calibrated thresholds run slightly high on phones (mic + AGC + VOICE_RECOGNITION processing shift scores down). Expect a small margin loss.

| On-device symptom | Meaning | Action |
|---|---|---|
| Heartbeat maxScore just below threshold when speaking the wake word | Domain shift, model fine | Lower the slider ~0.05–0.1; plan phone-mic training data |
| Heartbeat maxScore near 0 when speaking | Model not detecting at all | Check mic permission/source; re-check report score distributions |
| False fires on TV/speech | Real-world hard negatives | Record those exact sounds as NEG·train with the recorder app |
| No heartbeats at all while talking | Energy VAD gating or capture failure | Speak louder/closer; check `adb logcat -s WakeWordDetector` for errors |

## Slider vs metadata

The slider is live demo tuning and persists for the app session. The durable value is `threshold` in the run's `model_metadata.json` (calibrated on validation). If a slider value proves right on device, re-calibrate by adjusting `quantization.target_validation_fpr` in `wakeword.yaml` and re-export — don't hand-edit assets as a permanent fix.

## Never claim device performance without measuring

No device CI exists. Latency/power claims require an actual on-device benchmark (`WakeWordModelBenchmark`). Metrics from `evaluate` are desktop numbers.
