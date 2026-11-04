# Contributing

Install `pip install -e '.[dev]'`, run `ruff check .` and `pytest`. Training changes require `.[train]`; Android changes require JDK 17 and Android SDK 35. Add regression tests for DSP, split isolation, calibration and metrics. Never tune thresholds on test data. Never commit user recordings or generated speech without explicit rights and consent.
