from pathlib import Path
import csv,json
import numpy as np
from scipy.io import wavfile
from .audio import mono_resample,sliding_windows
from .frontend import log_mel
from .split import group_split

def read_wav(path):
    sr,x=wavfile.read(path)
    if x.dtype.kind in 'iu': x=x.astype(np.float32)/max(abs(np.iinfo(x.dtype).min),np.iinfo(x.dtype).max)
    return mono_resample(x,int(sr))

def scan(project):
    p=Path(project); rows=[]
    for label in ('positive','negative'):
        for f in sorted((p/'data/raw'/label).glob('*.wav')): rows.append({'path':str(f),'label':int(label=='positive'),'source_group':f.stem.split('__')[0]})
    meta=p/'data/raw/metadata.csv'
    if meta.exists():
        with meta.open() as fh:
            for row in csv.DictReader(fh):
                for item in rows:
                    if Path(item['path']).name==Path(row['path']).name: item['source_group']=row['speaker_id']
    return rows

def prepare(project,seed=42):
    p=Path(project); rows=scan(p)
    if not rows: raise ValueError('No WAVs found under data/raw/{positive,negative}')
    splits=group_split(rows,seed); out=p/'data/processed'; out.mkdir(parents=True,exist_ok=True)
    for name,items in splits.items():
        xs=[]; ys=[]
        for item in items:
            for w in sliding_windows(read_wav(item['path'])):
                f=log_mel(w)
                if f.shape[0]<98: f=np.pad(f,((0,98-f.shape[0]),(0,0)))
                xs.append(f[:98]); ys.append(item['label'])
        np.savez_compressed(out/f'{name}.npz',x=np.asarray(xs,np.float32),y=np.asarray(ys,np.int64))
    (out/'split_manifest.json').write_text(json.dumps({k:[r['path'] for r in v] for k,v in splits.items()},indent=2))
    return splits
