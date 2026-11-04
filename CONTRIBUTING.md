# Contributing

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,train]'
ruff check .
timeout 90s pytest -q
```

Android tests/build:

```bash
gradle -p android :wakeword-sdk:testDebugUnitTest :demo:assembleDebug --no-daemon
```

## Data and evaluation rules

- Do not commit recordings, generated speech, model files, processed datasets, private metadata or speaker-identifying information.
- Split by stable source/speaker groups before extracting windows or augmenting.
- Augmentation backgrounds must come from the training partition only.
- Validation is for threshold/calibration decisions; test data is a one-time final check.
- Use continuous audio from unseen rooms/devices and report false activations/hour; window accuracy alone is not sufficient.

## Frontend contract

Any change to the Python frontend must update and explain the Kotlin implementation, regenerate `android/wakeword-sdk/src/test/resources/frontend_golden.csv` with the script, and pass the Kotlin golden-value test. Model metadata must include exact input/output quantization parameters and normalization arrays.

## Pull requests

Add tests for preprocessing, splitting, augmentation, export metadata, stream metrics or Android lifecycle changes. Long-running tests must have explicit timeouts. Never claim a target-device latency/power result without a device benchmark and methodology.
