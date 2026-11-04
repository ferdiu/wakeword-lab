def build_model(input_shape=(98,40,1),width=32,dropout=.15,label_smoothing=.05):
    import tensorflow as tf
    L=tf.keras.layers; inp=L.Input(input_shape); x=L.Conv2D(width,3,strides=2,padding='same',use_bias=False)(inp); x=L.BatchNormalization()(x); x=L.ReLU()(x)
    for c,s in ((width,1),(width*2,2),(width*2,1),(width*3,2),(width*3,1)):
        x=L.DepthwiseConv2D(3,strides=s,padding='same',use_bias=False)(x); x=L.BatchNormalization()(x); x=L.ReLU()(x)
        x=L.Conv2D(c,1,use_bias=False)(x); x=L.BatchNormalization()(x); x=L.ReLU()(x)
    x=L.GlobalAveragePooling2D()(x); x=L.Dropout(dropout)(x); out=L.Dense(1,activation='sigmoid')(x)
    m=tf.keras.Model(inp,out); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),metrics=['accuracy']); return m
