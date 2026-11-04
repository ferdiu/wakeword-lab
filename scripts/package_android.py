from pathlib import Path
import argparse
import shutil

p = argparse.ArgumentParser(description="Copy trained INT8 model and metadata into Android demo assets")
p.add_argument("project")
p.add_argument("--run-name", default="default")
p.add_argument("--destination", default="android/demo/src/main/assets")
a = p.parse_args()
src = Path(a.project) / "artifacts/experiments" / a.run_name
dst = Path(a.destination); dst.mkdir(parents=True, exist_ok=True)
for name in ("wakeword_int8.tflite", "model_metadata.json"):
    shutil.copy2(src / name, dst / name)
print(dst.resolve())
