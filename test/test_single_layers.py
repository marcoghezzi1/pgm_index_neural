import unittest
from custom_layer import myCustom
from pgm_layer import custom_pgm
from modello import pgm_index
import numpy as np
from keras.layers import Input
from keras.models import Model
import tensorflow as tf

w = np.array([1, 8]).reshape(1, 2)
end = np.array([8, 20]).reshape(1, 2)
slope = np.array([5, 3]).reshape(1, 2)
intercept = np.array([3, -4]).reshape(1, 2)


class MyTestCaseCustomLayer(unittest.TestCase):
    def test_bin1_custom_layer(self):
        data = Input(shape=(1,), dtype=np.float64)
        custom = myCustom(units=2, init=w, train=False, end=end, dtype=np.float64)
        out = custom(data)
        model = Model(inputs=data, outputs=out)
        x = model.predict([10])
        y = np.array([[0, 1]])
        self.assertTrue((x == y).all())

    def test_bin2_custom_layer(self):
        data = Input(shape=(1,), dtype=np.float64)
        custom = myCustom(units=2, init=w, train=False, end=end, dtype=np.float64)
        out = custom(data)
        model = Model(inputs=data, outputs=out)
        x = model.predict([7])
        y = np.array([[1, 0]])
        self.assertTrue((x == y).all())

    def test_pgm_layer(self):
        pgm = custom_pgm(2, w, slope, intercept, True, name='custom', dtype=np.float64)
        print(pgm.trainable)
        data = Input(shape=(1,), dtype=np.float64)
        ris = pgm(data)
        model = Model(inputs=data, outputs=ris)
        print(model.summary())
        x = model.predict(np.array([10], dtype=np.float64))
        print(x)
        y = np.array([[48, 2]])
        self.assertTrue((x == y).all())

    def test_pgm(self):
        model = pgm_index(2, w, end, slope, intercept, False)
        #print(inputs)
        x = model.predict([10])
        print(x)
        y = np.array([[2]])
        self.assertTrue((x == y).all())


if __name__ == '__main__':
    unittest.main()
