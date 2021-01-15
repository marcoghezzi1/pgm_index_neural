from keras.layers import Layer
from keras.initializers import Initializer

import numpy as np
import tensorflow as tf


class my_init(Initializer):
    def __init__(self, w):  # w è il vettore dei pesi con cui inizializzare
        self.w = w

    def __call__(self, shape, dtype=np.float64):
        return self.w

    def get_config(self):
        return {'w': self.w}


class custom_pgm(Layer):
    def __init__(self, units, init, slope, intercept, train, **kwargs):
        super(custom_pgm, self).__init__(**kwargs)
        self.units = units
        self.initW = init
        self.trainable = train
        self.slope = slope
        self.intercept = intercept
        self.initializer_slope = tf.keras.initializers.RandomUniform(minval=0., maxval=1e-4)
        self.initializer_intercept = tf.keras.initializers.RandomUniform(minval=-2, maxval=2)

    def build(self, input_shape):
        self.init = self.add_weight(name='kernel',
                                    shape=(input_shape[1], self.units),
                                    initializer=my_init(self.initW), trainable=False, dtype=np.float64)
        self.slope = self.add_weight(name='kernel', shape=(input_shape[1], self.units), initializer=my_init(self.slope), trainable=self.trainable, dtype=np.float64)
        self.intercept = self.add_weight(name='kernel', shape=(input_shape[1], self.units), initializer=my_init(self.intercept), trainable=self.trainable, dtype=np.float64)

    def call(self, inputs):
        diff = inputs - self.init
        prod = tf.cast(diff, np.float64) * self.slope
        prod += self.intercept
        return prod

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units
