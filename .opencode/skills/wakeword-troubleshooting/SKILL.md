---
name: wakeword-troubleshooting
description: Use when a wakeword-lab command or Android build fails or behaves unexpectedly — Edge TTS synthesis errors, prepare split errors, training stuck at low validation accuracy, exported model scoring everything low, on-device misses or no notification, Gradle/JVM build errors, adb pull failures. Symptom-to-fix table for the known failure modes.
---

# Wakeword-lab troubleshooting

## Data / synthesis

| Symptom | Cause | Fix |
|---|---|---|
| `prepare` fails: split needs positive and negative | All data pinned/auto-assigned to train | Record real takes pinned to validation/test (`record --split` or recorder app) |
| Edge TTS `NoAudioReceived` aborts | Transient service drop | Already handled (retry then skip). Re-run to top up if many were skipped |
| `ffmpeg executable not found` | Missing dependency | Install ffmpeg with the OS package manager |
| `Install the TTS/record extra` | Optional extra not installed | `pip install -e '.[tts]'` / `pip install -e '.[record]'` |
| fetch-negatives/fetch-noise SHA-256 mismatch | Corrupt/swapped download | Re-run; the check fails closed on purpose. Check network/proxy |

## Training / metrics

| Symptom | Cause | Fix |
|---|---|---|
| Train acc ~99%, validation stuck ~35% | TTS-only training doesn't generalize | Real voice recordings in train split |
| `Only N validation negatives: empirical FPR resolution...` warning + recall 0 at threshold | Too few validation negatives; threshold = max negative score | Add validation negatives (real speech/ambient) |
| Exported model scores everything low | Check float vs int8 drift in report; if drift tiny, the model itself is weak | More data / regularization — see wakeword-training-cycle |
| Test recall suddenly drops after data change | Split leakage or group moved across splits | Inspect `data/processed/split_manifest.json` groups per split; fix metadata pins, re-prepare |

## Android

| Symptom | Cause | Fix |
|---|---|---|
| `Inconsistent JVM-target compatibility` / gradle rejects Java | System Java too new (e.g. 25) | `export JAVA_HOME=~/.gradle/jdks/eclipse_adoptium-17-amd64-linux.2` (or any JDK 17) |
| `gradle: command not found` | No system gradle | Use `android/gradlew` (wrapper, 8.13) |
| App stuck on "Preparing offline wake-word model…", no notification | POST_NOTIFICATIONS denied (Android 13+) | Tap Start again and grant; or Settings → Apps → notifications |
| Green mic dot but no reaction | Detections only surface via notification/log | Open the app — the log view shows heartbeats; or `adb logcat -s WakeWordDetector` |
| Near-miss on device (score just under threshold) | Phone mic domain shift | Lower slider; collect phone-mic training data |
| `adb pull` finds nothing in fetch-recordings | No recordings yet / wrong device state | Record takes first; check `adb devices` shows the phone |
| `adb: command not found` in fetch-recordings | adb not on PATH | Set ANDROID_HOME or install platform-tools (`~/.android-sdk/sdk/platform-tools/adb` is auto-detected) |

## Golden rule for new failures

Reproduce with the smallest command, check the report/logs before changing code, and fix the root cause — do not tune thresholds or config to mask a data problem.
