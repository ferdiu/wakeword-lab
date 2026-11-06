# AGENTS.md

Wake-word training/quantization lab: Python pipeline (`src/wakeword_lab/`, CLI `wakeword-lab`) plus an Android SDK + demo (`android/`). See `README.md` for the full workflow and `CONTRIBUTING.md` for data/eval rules — both are accurate; follow them.

## Commands

```bash
pip install -e '.[dev,train]'   # 'train' (TensorFlow/librosa) and 'tts' extras are separate; core deps are numpy/scipy/PyYAML only
ruff check .
timeout 90s pytest -q                                  # full unit suite must finish fast; CI enforces a 90 s timeout
python -m pytest tests/test_tf_integration.py          # TF training/export smoke; needs [train], CI gives it 15m
gradle -p android :wakeword-sdk:testDebugUnitTest :wakeword-sdk:assembleRelease :demo:assembleDebug --no-daemon
```

- Lint config in `pyproject.toml` (ruff, line-length 100, E501 ignored). pytest `pythonpath = ["src"]` — no install needed to run tests.
- Android toolchain: JDK 17, Gradle 8.13, AGP 8.13.2, Android SDK Platform/Build-Tools 36. CI is the source of truth (`.github/workflows/`). A Gradle wrapper (`android/gradlew`, 8.13) exists; JDK 17 must be active (`export JAVA_HOME=~/.gradle/jdks/eclipse_adoptium-17-amd64-linux.2` on this machine — system Java is 25 and AGP rejects it). SDK lives at `~/.android-sdk/sdk` via gitignored `android/local.properties`.

## Non-obvious constraints

- **Python/Kotlin frontend parity**: any change to `src/wakeword_lab/frontend.py` (log-mel) must be mirrored in the Kotlin SDK, then regenerate the golden fixture with `python scripts/generate_frontend_fixture.py` (writes `android/wakeword-sdk/src/test/resources/frontend_golden.csv`) and pass both test suites. This is a hard contract.
- **Never commit data or models**: recordings, WAVs, `.npz`, `.tflite`, `metadata.csv`, `artifacts/`, and `android/demo/src/main/assets/model_metadata.json` are gitignored on purpose. Do not force-add them.
- **Split integrity**: split by source/speaker group *before* windowing; augmentation uses training-partition audio only; validation is for threshold calibration, test is a one-time final check. Each split must contain both positive and negative examples (`prepare` fails otherwise).
- **Timeouts**: long-running tests/commands must have explicit timeouts (CI uses `timeout` everywhere).
- No physical-device CI: never state device latency/power numbers without an actual on-device benchmark.
- `vox-wakeword/` is a local user project (data + config + artifacts), not library code — treat its contents as private data.
- TTS synthesis (`wakeword-lab synthesize`) and `wakeword-lab fetch-negatives` are the only networked steps (Edge TTS + requires `ffmpeg`; mini_speech_commands download); everything else is local-only.
- `wakeword-lab record` (mic capture) needs the `record` extra: `pip install -e '.[record]'` (sounddevice + soundfile).
- Model deployment: `wakeword-lab package-android ./project --run-name X` copies `wakeword_int8.tflite` + `model_metadata.json` into the demo app assets; metadata quantization params must exactly match the exported model.
