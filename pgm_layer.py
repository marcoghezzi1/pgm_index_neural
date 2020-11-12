from keras.layers import Layer
from keras.initializers import Initializer

import numpy as np
from keras import backend as K


class my_init(Initializer):
    def __init__(self, w):  # w è il vettore dei pesi con cui inizializzare
        self.w = w

    def __call__(self, shape, dtype=np.float32):
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

    def build(self, input_shape):
        self.init = self.add_weight(name='kernel',
                                    shape=(input_shape[1], self.units),
                                    initializer=my_init(self.initW), trainable=self.trainable)
        self.slope = self.add_weight(name='kernel',
                                     shape=(input_shape[1], self.units),
                                     initializer=my_init(self.slope), trainable=self.trainable)
        self.intercept = self.add_weight(name='kernel',
                                         shape=(input_shape[1], self.units),
                                         initializer=my_init(self.intercept), trainable=self.trainable)
        super(custom_pgm, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        diff = inputs - self.init
        prod = diff * self.slope
        prod += self.intercept
        return prod

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units