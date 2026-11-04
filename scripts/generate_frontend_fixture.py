from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wakeword_lab.frontend import log_mel

sample_rate = 16000
t = np.arange(sample_rate, dtype=np.float64) / sample_rate
wave = (0.10 * np.sin(2 * np.pi * 440.0 * t) + 0.035 * np.sin(2 * np.pi * 1000.0 * t + 0.2)).astype(np.float32)
features = log_mel(wave)
path = Path(__file__).resolve().parents[1] / "android/wakeword-sdk/src/test/resources/frontend_golden.csv"
np.savetxt(path, features, delimiter=",", fmt="%.9g")
print(f"Wrote {features.shape} to {path}")
