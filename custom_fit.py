import tensorflow as tf
from tensorflow import keras
from pgm_layer import custom_pgm
from custom_layer import myCustom
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Multiply
from sys import maxsize
import numpy as np


class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        #tf.print(trainable_vars)
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def pgm_index(neuroni, seg_init, seg_end, slopes, intercepts, trainable):
    # layer input
    data = Input(shape=(1,), dtype=np.float64)          # dtype = np.float64

    # layer left
    custom = myCustom(units=neuroni, init=seg_init, train=False, end=seg_end, dtype=np.float64, name="left")
    left = custom(data)

    # layer right
    pgm = custom_pgm(neuroni, seg_init, slopes, intercepts, trainable, dtype=np.float64, name="right")
    right = pgm(data)

    # mult layer
    mul = Multiply(trainable=False, dtype=np.float64, name="mul")([left, right])

    # output layer
    dense2 = Dense(1, kernel_initializer='ones', use_bias=False, trainable=False, dtype=np.float64, name="dense")             # input_dtype=np.float64
    out = dense2(mul)
    model = CustomModel(inputs=data, outputs=out)
    return model


# --- test ---
w = np.array([1, 8, 20, 25, 34, 39]).reshape(1, 6)
end = np.array([8, 20, 25, 34, 39, maxsize]).reshape(1, 6)
slope = np.array([5, 3, 10, 27, 2, 9]).reshape(1, 6)
intercept = np.array([3, -4, 7, -10, 13, -14]).reshape(1, 6)
neuroni = 6

pgm = pgm_index(neuroni, w, end, slope, intercept, True)
x = np.array([2, 4, 10, 10, 13, 15, 17, 21, 21, 24, 25, 36, 38, 42])
y = np.array([0, 1, 2, 2, 4, 5, 6, 7, 7, 9, 10, 11, 12, 13])
# print(pgm.summary())
pgm.compile(loss='mae', optimizer='adam')
pgm.fit(x, y, epochs=10000)
print(pgm.get_weights())




