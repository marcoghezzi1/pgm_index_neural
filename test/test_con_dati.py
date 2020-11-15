import unittest

import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from modello import pgm_index


with open('../datipisa/segments_wiki_ts_1M_uint64.csv') as file:
    indice = pd.read_csv(file, header=2)
dati = np.fromfile("../datipisa/wiki_ts_1M_uint64", dtype=np.uint64)

dati = dati.reshape(len(dati), 1)
init = indice['key']
slope = indice['slope']
intercept = indice['intercept']
end = indice['key'][1:].to_numpy()
end = np.append(end, sys.maxsize)
neuroni = len(indice)


class MyTestCase(unittest.TestCase):
    def test_something(self):
        pgm = pgm_index(neuroni, init, end, slope, intercept, False)
        y = pgm.predict(dati).reshape(-1, 1)
        index = [0]
        for i in range(1, len(dati)):
            if dati[i] == dati[i - 1]:
                index.append(index[i - 1])
            else:
                index.append(i)
        err = []

        for i in range(len(dati)):
            diff = abs(y[i, 0] - index[i])
            err.append(diff)

        x = all(i <= 64.1 for i in err)
        self.assertEqual(True, x)


if __name__ == '__main__':
    unittest.main()
