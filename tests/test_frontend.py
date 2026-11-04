import numpy as np
from wakeword_lab.frontend import log_mel

def test_shape_finite():
 f=log_mel(np.zeros(16000,np.float32)); assert f.shape==(98,40); assert np.isfinite(f).all()
def test_deterministic():
 x=np.random.default_rng(5).normal(size=16000).astype(np.float32); assert np.array_equal(log_mel(x),log_mel(x))
