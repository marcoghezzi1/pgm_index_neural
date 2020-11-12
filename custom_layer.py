from keras.layers import Layer
from keras.initializers import Initializer

import numpy as np
from keras import backend as K
import tensorflow as tf


class my_init(Initializer):
    def __init__(self, w):  # w è il vettore dei pesi con cui inizializzare
        self.w = w

    def __call__(self, shape, dtype=np.float32):
        return self.w

    def get_config(self):
        return {'w': self.w}


class myCustom(Layer):
    def __init__(self, units, init, train, end, **kwargs):
        super(myCustom, self).__init__(**kwargs)
        self.units = units
        self.initW = init
        self.trainable = train
        self.initB = end

    def build(self, input_shape):
        self.init = self.add_weight(name='kernel',
                                    shape=(input_shape[1], self.units),
                                    initializer=my_init(self.initW), trainable=self.trainable)
        self.end = self.add_weight(name='kernel',
                                   shape=(input_shape[1], self.units),
                                   initializer=my_init(self.initB), trainable=self.trainable)
        super(myCustom, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        a = K.greater_equal(inputs, self.init)
        b = K.greater(self.end, inputs)
        y = K.ones(K.shape(inputs))

        out = tf.where(K.all(K.stack([a, b], axis=0), axis=0), K.ones(K.shape(inputs)), K.zeros(K.shape(inputs)))

        # K.all applica AND alla matrice K.stack
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units
