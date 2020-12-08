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
        self.prova = np.array([[sys.maxsize]], dtype=np.float64)

    def build(self, input_shape):
        self.init = self.add_weight(name='kernel',
                                    shape=(input_shape[1], self.units),
                                    initializer=my_init(self.initW), trainable=self.trainable, dtype=np.float64)
        self.x = tf.concat((self.init, self.prova), axis=-1)

        self.x = tf.roll(self.x, shift=[0, -1], axis=[0, 1])
        self.x = self.x[:, :-1]

    def call(self, inputs):
        a = K.greater_equal(inputs, self.init)
        b = K.greater(self.x, inputs)
        out = float(K.all(K.stack([a, b], axis=0), axis=0))
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units