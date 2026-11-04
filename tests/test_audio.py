import numpy as np
from wakeword_lab.audio import mono_resample,fit_window,sliding_windows

def test_resample(): assert len(mono_resample(np.zeros(8000),8000))==16000
def test_stereo(): assert np.allclose(mono_resample(np.c_[np.ones(10),-np.ones(10)],16000),0)
def test_padding(): assert np.array_equal(fit_window([1,2],4),[1,2,0,0])
def test_sliding(): assert len(list(sliding_windows(np.zeros(32000),16000,8000)))==3


def test_multichannel_audio_downmix_uses_channel_axis_for_more_than_eight_channels():
    channels = np.linspace(-0.5, 0.4, 10, dtype=np.float32)
    samples = np.tile(channels, (100, 1))
    mono = mono_resample(samples, 16000)
    assert mono.shape == (100,)
    assert np.allclose(mono, channels.mean())
