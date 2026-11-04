import numpy as np
import pytest
from wakeword_lab.dataset import _positive_clips

def test_unannotated_long_positive_uses_single_centered_crop():
    audio = np.zeros(40_000, np.float32)
    audio[12_000:28_000] = 0.25
    clips = _positive_clips(audio, np.random.default_rng(1), 80)
    assert len(clips) == 1
    assert clips[0].shape == (16_000,)
    assert np.all(clips[0] == 0.25)

def test_timestamped_positive_windows_all_contain_complete_phrase():
    audio = np.zeros(40_000, np.float32)
    audio[12_000:20_000] = 0.3
    clips = _positive_clips(audio, np.random.default_rng(2), 5, 0.75, 1.25)
    assert 1 <= len(clips) <= 5
    assert all(np.count_nonzero(clip) == 8_000 for clip in clips)

def test_rejects_phrase_longer_than_model_context():
    audio = np.zeros(32_000, np.float32)
    with pytest.raises(ValueError, match="one-second"):
        _positive_clips(audio, np.random.default_rng(0), 5, 0.1, 1.2)

@pytest.mark.parametrize("start,end", [(-0.1, 0.2), (0.4, 0.2), (float("nan"), 0.5), (0.1, 1.1)])
def test_short_recording_still_validates_phrase_timestamps(start, end):
    audio = np.zeros(12_000, np.float32)
    with pytest.raises(ValueError):
        _positive_clips(audio, np.random.default_rng(0), 5, start, end)
