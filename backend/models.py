"""Four deep-learning classifiers sharing the same inputs: numeric features + station embedding."""
from .config import CLASSES, EMB_DIM, MODEL_SPECS


def build_model(key, n_num, n_stations, n_classes=len(CLASSES)):
    import tensorflow as tf
    from tensorflow.keras import Model, layers
    num = layers.Input(shape=(n_num,), name="num")
    st = layers.Input(shape=(1,), dtype="int32", name="station")
    emb = layers.Flatten()(layers.Embedding(n_stations + 1, EMB_DIM)(st))
    x = layers.Concatenate()([num, emb])                                  # vector of n_num + EMB_DIM values
    d = n_num + EMB_DIM
    if key == "ann":
        x = layers.Dropout(0.3)(layers.Dense(128, activation="relu")(x))
        x = layers.Dropout(0.2)(layers.Dense(64, activation="relu")(x))
        x = layers.Dense(32, activation="relu")(x)
    elif key == "dnn":
        x = layers.Dense(128)(x)
        for _ in range(3):
            h = layers.ReLU()(layers.BatchNormalization()(layers.Dense(128)(x)))
            h = layers.BatchNormalization()(layers.Dense(128)(h))
            x = layers.Dropout(0.2)(layers.ReLU()(layers.Add()([x, h])))
    elif key == "cnn":
        x = layers.Reshape((d, 1))(x)
        x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.2)(layers.Dense(64, activation="relu")(x))
    elif key == "gru":
        x = layers.GRU(48)(layers.Reshape((d, 1))(x))
        x = layers.Dense(32, activation="relu")(x)
    else:
        raise ValueError(key)
    out = layers.Dense(n_classes, activation="softmax", name="cls")(x)
    model = Model([num, st], out, name=MODEL_SPECS[key][0].replace(" ", "_"))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
