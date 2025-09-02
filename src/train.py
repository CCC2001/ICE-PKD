import os
import tensorflow as tf
from .losses import kl_divergence, wass_loss

class Trainer:
    def __init__(self, cfg):
        self.cfg  = cfg
        self.opt  = tf.keras.optimizers.Adam(cfg.lr)
        self.bce  = tf.keras.losses.BinaryCrossentropy()
        self.cce  = tf.keras.losses.CategoricalCrossentropy()

        self.corrector = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(cfg.l2),
                                  input_shape=(2,)),
            tf.keras.layers.Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(cfg.l2)),
            tf.keras.layers.Dense(3, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(cfg.l2))
        ])

    def compile(self, model, model0):
        self.model  = model
        self.model0 = model0  # frozen teacher

    @tf.function
    def train_step(self, x, y, t, x_pre, t_pre):
        cfg = self.cfg
        with tf.GradientTape(persistent=True) as tape:
            out, est_t, w, gamma = self.model(x, training=True)
            yf = tf.reduce_sum(out * t, 1, keepdims=True)

            # sample weights
            n_t = tf.reduce_sum(t, 0)
            r   = n_t / tf.reduce_sum(n_t)
            w1  = tf.reduce_sum(t * r, 1, keepdims=True)
            w2  = 1 / tf.reduce_sum(w * t, 1, keepdims=True)
            weights = w1 * w2

            loss_bce = self.bce(y, yf, sample_weight=tf.squeeze(weights))

            # regularization / constraints
            constraint = cfg.lambd * tf.reduce_sum(
                tf.nn.relu(out[:,:-1] - out[:,1:]))

            loss_wass = wass_loss(gamma, t, cfg.lambd)

            loss_pi = cfg.beta * self.cce(t, est_t)

            # ATE
            bcr_pred = tf.reduce_sum(yf*t, 0)/tf.reduce_sum(t,0)
            bcr_true = tf.reduce_sum(tf.reshape(y,[-1,1])*t,0)/tf.reduce_sum(t,0)
            ate_loss = .01*tf.reduce_sum(tf.abs(bcr_true-bcr_pred))

            loss0 = loss_bce + constraint + loss_wass + loss_pi + ate_loss

            # distillation
            out_pre = self.model(x_pre)[0]
            out_pre_c = self.corrector(x_pre)
            out0_pre = self.model0(x_pre, training=False)[0]
            loss1 = kl_divergence(out0_pre + out_pre_c, out_pre)

            loss = loss0 + .8*loss1

        grads  = tape.gradient(loss, self.model.trainable_variables)
        grads_c= tape.gradient(loss, self.corrector.trainable_variables)
        self.opt.apply_gradients(zip(grads,  self.model.trainable_variables))
        self.opt.apply_gradients(zip(grads_c,self.corrector.trainable_variables))
        return loss

    def fit(self, train_ds, val_ds, x_pre, t_pre, epochs, ckpt_path):
        best_auc = 0
        for ep in range(epochs):
            for step, ((x,y,t),_) in enumerate(train_ds):
                loss = self.train_step(x, y, t, x_pre, t_pre)
                if step % 20 == 0:
                    print(f"Epoch {ep} step {step} loss={loss:.4f}")

            # validation
            y_true, y_pred = [], []
            for (x,t),y in val_ds:
                p = self.model(x, training=False)[0]
                y_hat = tf.reduce_sum(p*t, 1)
                y_true.append(y)
                y_pred.append(y_hat)
            y_true = tf.concat(y_true,0)
            y_pred = tf.concat(y_pred,0)

            auc = tf.keras.metrics.AUC()(y_true, y_pred).numpy()
            print(f"Epoch {ep}  val_auc={auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                self.model.save_weights(ckpt_path)
        return best_auc