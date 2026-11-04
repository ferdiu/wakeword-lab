from pathlib import Path
import json
import numpy as np

def train(project,epochs=40,batch_size=64):
    import tensorflow as tf
    from .model import build_model
    p=Path(project); d=p/'data/processed'; tr=np.load(d/'train.npz'); va=np.load(d/'validation.npz')
    mean=tr['x'].mean((0,1)); std=tr['x'].std((0,1))+1e-5
    norm=lambda x: ((x-mean)/std)[...,None]
    (p/'artifacts').mkdir(exist_ok=True); model=build_model()
    model.fit(norm(tr['x']),tr['y'],validation_data=(norm(va['x']),va['y']),epochs=epochs,batch_size=batch_size,callbacks=[tf.keras.callbacks.EarlyStopping(patience=7,restore_best_weights=True)])
    model.save(p/'artifacts/model.keras'); (p/'artifacts/normalization.json').write_text(json.dumps({'mean':mean.tolist(),'std':std.tolist()})); return model
