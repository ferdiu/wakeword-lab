import argparse,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(prog='wakeword-lab'); sub=ap.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('init'); q.add_argument('project'); q.add_argument('--wake-word',default='Hey Nova')
    for cmd in ('prepare','train','export','evaluate','report'):
        q=sub.add_parser(cmd); q.add_argument('project')
    x=ap.parse_args(); p=Path(getattr(x,'project','.'))
    if x.cmd=='init':
        for d in ('data/raw/positive','data/raw/negative','data/raw/noise','data/processed','artifacts'): (p/d).mkdir(parents=True,exist_ok=True)
        (p/'wakeword.json').write_text(json.dumps({'wake_word':x.wake_word},indent=2)); return
    if x.cmd=='prepare':
        from .dataset import prepare; prepare(p)
    elif x.cmd=='train':
        from .train import train; train(p)
    elif x.cmd=='export':
        from .export import export; export(p)
    elif x.cmd=='evaluate':
        import tensorflow as tf
        from .metrics import classification_metrics
        d=np.load(p/'data/processed/test.npz'); n=json.loads((p/'artifacts/normalization.json').read_text()); m=json.loads((p/'artifacts/model_metadata.json').read_text())
        model=tf.keras.models.load_model(p/'artifacts/model.keras'); z=(d['x']-np.array(n['mean']))/np.array(n['std']); scores=model.predict(z[...,None],verbose=0).ravel()
        (p/'artifacts/evaluation.json').write_text(json.dumps(classification_metrics(scores,d['y'],m['threshold']),indent=2))
    elif x.cmd=='report':
        for f in sorted((p/'artifacts').glob('*.json')): print(f'## {f.name}\n{f.read_text()}')
if __name__=='__main__': main()
