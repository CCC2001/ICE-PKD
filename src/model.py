import tensorflow as tf
from tensorflow.keras import layers, regularizers

def build(d, cfg):
    inp = tf.keras.Input(shape=(d,))
    l2  = cfg.l2
    repr_units = cfg.repr_units

    # factors
    factors = []
    for _ in range(3):
        x = inp
        for _ in range(cfg.repr_layers):
            x = layers.Dense(repr_units, activation='relu',
                             kernel_regularizer=regularizers.l2(l2))(x)
            x = layers.BatchNormalization()(x)
        factors.append(x)
    beta, delta, gamma = factors

    # heads
    x = tf.concat([delta, gamma], 1)
    heads = []
    for _ in range(len(cfg.head_units)):
        h = x
        for u in cfg.head_units:
            h = layers.Dense(u//2, activation='relu',
                             kernel_regularizer=regularizers.l2(l2))(h)
            h = layers.BatchNormalization()(h)
            h = layers.Dense(u, activation='relu',
                             kernel_regularizer=regularizers.l2(l2))(h)
            h = layers.BatchNormalization()(h)
        heads.append(h)

    p = [layers.Dense(1, activation='sigmoid')(h) for h in heads]
    p_all = layers.concatenate(p, axis=1)

    # recovery
    rec = tf.concat([beta, delta], 1)
    for _ in range(cfg.repr_layers):
        rec = layers.Dense(repr_units, activation='relu',
                           kernel_regularizer=regularizers.l2(l2))(rec)
        rec = layers.BatchNormalization()(rec)
    est_t = layers.Dense(3, activation='softmax')(rec)

    # w
    w = delta
    for _ in range(cfg.repr_layers):
        w = layers.Dense(repr_units, activation='relu',
                         kernel_regularizer=regularizers.l2(l2))(w)
        w = layers.BatchNormalization()(w)
    w = layers.Dense(3, activation='softmax')(w)

    return tf.keras.Model(inp, [p_all, est_t, w, gamma])