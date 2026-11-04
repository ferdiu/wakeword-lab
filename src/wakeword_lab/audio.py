import numpy as np
from scipy.signal import resample_poly
from math import gcd

def mono_resample(samples, sample_rate, target_rate=16000):
    x=np.asarray(samples,dtype=np.float32)
    if x.ndim == 2:
        # Audio files are normally [frames, channels]; averaging the smaller dimension
        # also handles channel-first arrays and multichannel (>8) WAVs.
        x = x.mean(axis=1 if x.shape[1] <= x.shape[0] else 0)
    if x.ndim!=1 or sample_rate<=0 or target_rate<=0: raise ValueError("invalid audio or sample rate")
    if sample_rate!=target_rate:
        g=gcd(sample_rate,target_rate); x=resample_poly(x,target_rate//g,sample_rate//g).astype(np.float32)
    return np.clip(x,-1,1)

def fit_window(samples,length=16000,offset=0):
    if length<=0: raise ValueError("length must be positive")
    x=np.asarray(samples,dtype=np.float32); out=np.zeros(length,np.float32)
    chunk=x[max(0,offset):max(0,offset)+length]; out[:len(chunk)]=chunk
    return out

def sliding_windows(samples,length=16000,hop=3200):
    if length<=0 or hop<=0: raise ValueError("length/hop must be positive")
    x=np.asarray(samples,dtype=np.float32)
    if len(x)<length: yield fit_window(x,length); return
    for i in range(0,len(x)-length+1,hop): yield x[i:i+length].copy()
