import unittest
from custom_layer import myCustom
from pgm_layer import custom_pgm
from modello import pgm_index
import numpy as np
from keras.layers import Input
from keras.models import Model

w = np.array([1, 8]).reshape(1, 2)
end = np.array([8, 20])
slope = np.array([5, 3]).reshape(1, 2)
intercept = np.array([3, -4])


class MyTestCaseCustomLayer(unittest.TestCase):
    def test_bin1_custom_layer(self):
        data = Input(shape=(1,), dtype=np.int64)
        custom = myCustom(units=2, init=w, train=False, end=end, dtype=np.float64)
        out = custom(data)
        model = Model(inputs=data, outputs=out)
        x = model.predict([10])
        print(custom.dtype)
        y = np.array([[0, 1]])
        self.assertTrue((x == y).all())

    def test_bin2_customlayer(self):
        data = Input(shape=(1,))
        custom = myCustom(units=2, init=w, train=False, end=end)
        out = custom(data)
        model = Model(inputs=data, outputs=out)
        x = model.predict([7])
        y = np.array([[1, 0]])
        self.assertTrue((x == y).all())

    def test_pgm_layer(self):
        pgm_index = custom_pgm(2, w, slope, intercept, False)
        data = Input(shape=(1,))
        ris = pgm_index(data)
        model = Model(inputs=data, outputs=ris)
        x = model.predict([10, 2, 4])
        print(x)
        y = np.array([[23, 5]])
        self.assertTrue((x == y).all())

    def test_pgm(self):
        model = pgm_index(2, w, end, slope, intercept, False)
        inputs = np.array([[10, 2]])
        inputs = inputs.reshape(2, 1)
        print(inputs)
        x = model.predict(inputs)
        print(x)
        y = np.array([[2]])
        self.assertTrue((x == y).any)


if __name__ == '__main__':
    unittest.main()
