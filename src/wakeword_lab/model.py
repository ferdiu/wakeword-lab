
def keras_binary_smoothing(epsilon: float) -> float:
    """Map symmetric binary target smoothing to Keras BCE's smoothing convention.

    The configured epsilon means labels 0 -> epsilon and 1 -> 1-epsilon. Keras
    BinaryCrossentropy mixes each target with 0.5, so its parameter must be 2*epsilon.
    """
    import math
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or not 0.0 <= epsilon < 0.5:
        raise ValueError("binary label smoothing must be finite and in [0, 0.5)")
    return 2.0 * epsilon


def build_model(input_shape=(98, 40, 1), width=24, dropout=0.15,
                label_smoothing=0.05, learning_rate=0.001):
    """Small DS-CNN classifier. TensorFlow is imported lazily for light-weight tooling installs."""
    import tensorflow as tf
    layers = tf.keras.layers
    inputs = layers.Input(shape=input_shape, name="log_mel")
    x = layers.Conv2D(width, 3, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    for channels, stride in ((width, 1), (width * 2, 2), (width * 2, 1), (width * 3, 2), (width * 3, 1)):
        x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(channels, 1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="wakeword_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="wakeword_ds_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        # Keras mixes binary targets with 0.5. Doubling epsilon gives exactly
        # negative->epsilon and positive->1-epsilon (e.g. 0.05 / 0.95).
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=keras_binary_smoothing(label_smoothing)),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model
