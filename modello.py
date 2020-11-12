from custom_layer import myCustom
from pgm_layer import custom_pgm
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.layers import Multiply
import numpy as np


def pgm_index(neuroni, seg_init, seg_end, slopes, intercepts, trainable):
    # layer input
    data = Input(shape=(1,))

    # layer sinistro
    custom = myCustom(units=neuroni, init=seg_init, train=trainable, end=seg_end)
    sinistro = custom(data)

    # layer destro
    pgm = custom_pgm(neuroni, seg_init, slopes, intercepts, trainable)
    destro = pgm(data)

    # mult layer
    mul = Multiply()([sinistro, destro])

    # output layer
    dense2 = Dense(1, kernel_initializer='ones', use_bias=False, trainable=False)
    out = dense2(mul)
    model = Model(inputs=data, outputs=out)
    return model

