from custom_layer import myCustom
from pgm_layer import custom_pgm
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.layers import Multiply
import numpy as np


def pgm_index(neuroni, seg_init, slopes, intercepts, trainable):
    # layer input
    data = Input(shape=(1,), dtype=np.float64)

    # layer left
    custom = myCustom(units=neuroni, init=seg_init, train=False, dtype=np.float64, name="left")
    left = custom(data)

    # layer right
    pgm = custom_pgm(neuroni, seg_init, slopes, intercepts, trainable, dtype=np.float64, name="right")
    right = pgm(data)

    # mult layer
    mul = Multiply(trainable=False, dtype=np.float64, name="mul")([left, right])

    # output layer
    dense2 = Dense(1, kernel_initializer='ones', use_bias=False, trainable=False, dtype=np.float64, name="dense")
    out = dense2(mul)
    model = Model(inputs=data, outputs=out)
    return model

