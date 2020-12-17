from keras.layers import Layer
from keras.initializers import Initializer
import tensorflow as tf
import numpy as np
from keras import backend as K
import sys


class my_init(Initializer):
    def __init__(self, w):  # w è il vettore dei pesi con cui inizializzare
        self.w = w

    def __call__(self, shape, dtype=np.float64):
        return self.w

    def get_config(self):
        return {'w': self.w}


class myCustom(Layer):
    def __init__(self, units, init, train, **kwargs):
        super(myCustom, self).__init__(**kwargs)
        self.units = units
        self.initW = init
        self.trainable = train
        self.end = np.array([[sys.maxsize]], dtype=np.float64)

    def build(self, input_shape):
        self.init = self.add_weight(name='kernel',
                                    shape=(input_shape[1], self.units),
                                    initializer=my_init(self.initW), trainable=self.trainable, dtype=np.float64)

    def call(self, inputs):
        end = tf.concat((self.init, self.end), axis=-1)
        end = tf.roll(end, shift=[0, -1], axis=[0, 1])
        end = end[:, :-1]
        a = K.greater_equal(inputs, self.init)
        b = K.greater(end, inputs)
        c = K.cast(K.all(K.stack([a, b], axis=0), axis=0), np.float64)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units
