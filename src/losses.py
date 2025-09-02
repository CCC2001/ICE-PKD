import tensorflow as tf
from wass import wasserstein

def kl_divergence(y_true, y_pred):
    eps = tf.keras.backend.epsilon()
    y_true = tf.clip_by_value(y_true, eps, 1-eps)
    y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
    return tf.reduce_sum(
        tf.reduce_mean(y_true * tf.math.log(y_true/y_pred) +
                       (1-y_true) * tf.math.log((1-y_true)/(1-y_pred)), axis=0))

def wass_loss(gamma, t, lambd):
    loss = 0.0
    for m in range(1,3):
        loss += lambd * wasserstein(gamma, tf.reduce_sum(t*tf.range(3),1),
                                    .5, m, lam=10, its=10, sq=False, backpropT=False)
    return loss