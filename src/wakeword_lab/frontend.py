import numpy as np

def _mel(hz): return 2595*np.log10(1+np.asarray(hz)/700)

def mel_filterbank(sr=16000,n_fft=512,n_mels=40,fmin=60.,fmax=7600.):
    hz=700*(10**(np.linspace(_mel(fmin),_mel(fmax),n_mels+2)/2595)-1)
    bins=np.floor((n_fft+1)*hz/sr).astype(int); bank=np.zeros((n_mels,n_fft//2+1),np.float32)
    for m in range(1,n_mels+1):
        l,c,r=bins[m-1:m+2]
        if c>l: bank[m-1,l:min(c,bank.shape[1])]=np.arange(l,min(c,bank.shape[1]))-l
        if c>l: bank[m-1,l:min(c,bank.shape[1])]/=(c-l)
        if r>c:
            k=np.arange(max(0,c),min(r,bank.shape[1])); bank[m-1,k]=(r-k)/(r-c)
    return bank

def log_mel(samples,sample_rate=16000,frame_ms=25,hop_ms=10,n_fft=512,n_mels=40,fmin=60.,fmax=7600.):
    x=np.asarray(samples,dtype=np.float32)
    if x.ndim!=1 or sample_rate!=16000: raise ValueError("requires mono 16 kHz")
    frame=int(sample_rate*frame_ms/1000); hop=int(sample_rate*hop_ms/1000)
    if len(x)<frame: x=np.pad(x,(0,frame-len(x)))
    n=1+(len(x)-frame)//hop; idx=np.arange(frame)[None,:]+hop*np.arange(n)[:,None]
    frames=x[idx]*np.hanning(frame).astype(np.float32)[None,:]
    spec=np.fft.rfft(frames,n=n_fft,axis=1); power=(spec.real**2+spec.imag**2).astype(np.float32)
    return np.log(np.maximum(power@mel_filterbank(sample_rate,n_fft,n_mels,fmin,fmax).T,1e-6)).astype(np.float32)
