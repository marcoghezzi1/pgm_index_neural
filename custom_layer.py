from keras.layers import Layer
from keras.initializers import Initializer

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
        self.initB = np.append(init, sys.maxsize).reshape((1, self.units + 1))
        self.initB = np.roll(self.initB, shift=[0, -1], axis=[0, 1])
        self.initB = np.resize(self.initB, (1, self.units))

    def build(self, input_shape):
        self.init = self.add_weight(name='kernel',
                                    shape=(input_shape[1], self.units),
                                    initializer=my_init(self.initW), trainable=self.trainable, dtype=np.float64)
        self.end = self.add_weight(name='kernel',
                                   shape=(input_shape[1], self.units),
                                   initializer=my_init(self.initB), trainable=self.trainable, dtype=np.float64)

        super(myCustom, self).build(input_shape)

    def call(self, inputs):
        a = K.greater_equal(inputs, self.init)
        b = K.greater(self.end, inputs)
        out = float(K.all(K.stack([a, b], axis=0), axis=0))
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units
