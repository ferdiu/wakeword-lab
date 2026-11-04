import pytest
from wakeword_lab.stream import _intervals
from wakeword_lab.metrics import false_activations_per_hour

def test_intervals_accept_dict_and_pairs():
    assert _intervals([{"start":1,"end":2}, [3,4]]) == [(1.0,2.0),(3.0,4.0)]

def test_interval_rejects_empty_interval():
    with pytest.raises(ValueError): _intervals([[2,2]])

def test_event_times_must_be_in_duration():
    with pytest.raises(ValueError): false_activations_per_hour([9], 8)


def test_intervals_reject_non_finite_values():
    with pytest.raises(ValueError):
        _intervals([[float("nan"), 3]])


def test_stream_window_iterator_is_cadence_stable_across_arbitrary_block_boundaries():
    import numpy as np
    from wakeword_lab.stream import _iter_audio_windows

    audio = np.linspace(-0.5, 0.5, 16_640, dtype=np.float32)
    blocks = [audio[:777], audio[777:9_321], audio[9_321:15_555], audio[15_555:]]
    windows = list(_iter_audio_windows(blocks))
    assert [round(timestamp, 3) for _, _, timestamp in windows] == [1.0, 1.02, 1.04]
    assert np.array_equal(windows[0][0], audio[:16_000])
    assert np.array_equal(windows[1][0], audio[320:16_320])
    assert np.array_equal(windows[2][0], audio[640:16_640])


def test_stream_window_iterator_downmixes_stereo():
    import numpy as np
    from wakeword_lab.stream import _iter_audio_windows

    mono = np.full(16_000, 0.25, np.float32)
    stereo = np.column_stack((mono, -mono))
    windows = list(_iter_audio_windows([stereo]))
    assert len(windows) == 1
    assert np.allclose(windows[0][0], 0.0)
