import csv
import numpy as np
from scipy.io import wavfile
from wakeword_lab.dataset import scan, prepare

def _wav(path, sr=16000, value=.1):
    x = np.full(sr, value, np.float32)
    wavfile.write(path, sr, (x * 32767).astype(np.int16))

def test_metadata_group_and_split_pins(tmp_path):
    for label in ("positive", "negative", "noise"):
        (tmp_path / "data/raw" / label).mkdir(parents=True)
    for i in range(6):
        _wav(tmp_path / "data/raw/positive" / f"p{i}.wav", value=.1)
        _wav(tmp_path / "data/raw/negative" / f"n{i}.wav", value=.0)
    meta = tmp_path / "data/raw/metadata.csv"
    with meta.open("w", newline="") as f:
        w=csv.writer(f); w.writerow(["path","speaker_id","split","locale"])
        for i in range(6):
            w.writerow([f"p{i}.wav",f"speaker-{i}","train" if i==0 else "", "en-IN"])
            w.writerow([f"n{i}.wav",f"noise-{i}","", ""])
    rows=scan(tmp_path)
    p0=next(r for r in rows if r["path"].endswith("p0.wav"))
    assert p0["source_group"]=="speaker-0" and p0["split"]=="train"

def test_preparation_augments_train_only(tmp_path):
    for label in ("positive", "negative", "noise"):
        (tmp_path / "data/raw" / label).mkdir(parents=True)
    # Three positive + three negative groups, so every split receives both classes.
    for i in range(12):
        _wav(tmp_path / "data/raw/positive" / f"p{i}__speaker{i}.wav", value=.2)
        _wav(tmp_path / "data/raw/negative" / f"n{i}__room{i}.wav", value=.01)
    manifest=prepare(tmp_path, seed=1, augment=True)
    train=np.load(tmp_path/"data/processed/train.npz")
    val=np.load(tmp_path/"data/processed/validation.npz")
    test=np.load(tmp_path/"data/processed/test.npz")
    assert manifest["augmentation"]["train_only"] is True
    assert len(train["y"]) > len(val["y"]) and len(train["y"]) > len(test["y"])


def test_unsigned_8bit_wav_is_centered_around_zero(tmp_path):
    path = tmp_path / "unsigned.wav"
    signal = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    wavfile.write(path, 16000, signal)
    from wakeword_lab.dataset import read_wav
    audio = read_wav(path)
    assert audio[2] == 0.0
    assert audio[0] < 0 < audio[-1]


def test_noise_only_groups_default_to_train_and_do_not_fill_eval_splits(tmp_path):
    for label in ("positive", "negative", "noise"):
        (tmp_path / "data/raw" / label).mkdir(parents=True)
    for i in range(12):
        _wav(tmp_path / "data/raw/positive" / f"p{i}__speaker{i}.wav", value=.2)
        _wav(tmp_path / "data/raw/negative" / f"n{i}__room{i}.wav", value=.01)
    for i in range(8):
        _wav(tmp_path / "data/raw/noise" / f"amb{i}__ambient{i}.wav", value=.003)
    manifest = prepare(tmp_path, seed=5, augment=False)
    for split in ("validation", "test"):
        assert all(item["kind"] != "noise" for item in manifest["splits"][split])


def test_group_id_can_split_sessions_while_preserving_speaker_identity(tmp_path):
    from wakeword_lab.dataset import scan
    (tmp_path / "data/raw/positive").mkdir(parents=True)
    _wav(tmp_path / "data/raw/positive" / "one.wav")
    _wav(tmp_path / "data/raw/positive" / "two.wav")
    with (tmp_path / "data/raw/metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "speaker_id", "group_id", "split"])
        writer.writeheader()
        writer.writerow({"path": "one.wav", "speaker_id": "me", "group_id": "session-1", "split": "train"})
        writer.writerow({"path": "two.wav", "speaker_id": "me", "group_id": "session-2", "split": "test"})
    rows = scan(tmp_path)
    assert {row["source_group"] for row in rows} == {"session-1", "session-2"}
    assert {row["speaker_id"] for row in rows} == {"me"}


def test_noise_directory_contributes_train_only_negative_examples(tmp_path):
    for label in ("positive", "negative", "noise"):
        (tmp_path / "data/raw" / label).mkdir(parents=True)
    for i in range(12):
        _wav(tmp_path / "data/raw/positive" / f"p{i}__speaker{i}.wav", value=.2)
        _wav(tmp_path / "data/raw/negative" / f"n{i}__room{i}.wav", value=.01)
    _wav(tmp_path / "data/raw/noise" / "ambient__recording.wav", value=.003)
    manifest = prepare(tmp_path, seed=6, augment=False)
    train = np.load(tmp_path / "data/processed/train.npz")
    assert manifest["summary"]["train"]["negative"] >= 1
    for split in ("validation", "test"):
        assert all(row["kind"] != "noise" for row in manifest["splits"][split])
