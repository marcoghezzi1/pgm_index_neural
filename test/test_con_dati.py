import unittest
import pandas as pd
import numpy as np
from modello import pgm_index


def init_(segments, dati):
    global file, chiavi, init, slope, intercept, neuroni
    with open(segments) as file:
        indice = pd.read_csv(file, header=2)
    chiavi = np.fromfile(dati, dtype=np.uint64)
    chiavi = chiavi.reshape(len(chiavi), 1)
    init = indice['key'].to_numpy().reshape(1, len(indice))
    slope = indice['slope'].to_numpy().reshape(1, len(indice))
    intercept = indice['intercept'].to_numpy().reshape(1, len(indice))
    neuroni = len(indice)


class MyTestCase(unittest.TestCase):
    def test_wiki(self):
        init_('../datipisa/segments_wiki_ts_1M_uint64.csv', "../datipisa/wiki_ts_1M_uint64")
        pgm = pgm_index(neuroni, init, slope, intercept, False)
        y = pgm.predict(chiavi).reshape(-1, 1)
        index = [0]
        for i in range(1, len(chiavi)):
            if chiavi[i] == chiavi[i - 1]:
                index.append(index[i - 1])
            else:
                index.append(i)
        err = []

        for i in range(len(chiavi)):
            diff = abs(y[i, 0] - index[i])
            err.append(diff)
        x = all(i <= 64.1 for i in err)
        print(np.amax(err))
        self.assertEqual(True, x)

    def test_books(self):
        init_('../datipisa/segments_books_1M_uint64.csv', "../datipisa/books_1M_uint64")
        pgm = pgm_index(neuroni, init, slope, intercept, False)
        y = pgm.predict(chiavi).reshape(-1, 1)
        index = [0]
        for i in range(1, len(chiavi)):
            if chiavi[i] == chiavi[i - 1]:
                index.append(index[i - 1])
            else:
                index.append(i)
        err = []

        for i in range(len(chiavi)):
            diff = abs(y[i, 0] - index[i])
            err.append(diff)
        print(np.average(err))
        print(np.amax(err))
        x = all(i <= 64.1 for i in err)
        self.assertEqual(True, x)
        
    def test_longitude(self):
        init_('../datipisa/segments_longitude_1M_uint64.csv', "../datipisa/longitude_1M_uint64")
        pgm = pgm_index(neuroni, init, slope, intercept, False)
        y = pgm.predict(chiavi).reshape(-1, 1)
        index = [0]
        for i in range(1, len(chiavi)):
            if chiavi[i] == chiavi[i - 1]:
                index.append(index[i - 1])
            else:
                index.append(i)
        err = []

        for i in range(len(chiavi)):
            diff = abs(y[i, 0] - index[i])
            err.append(diff)

        x = all(i <= 64.1 for i in err)
        self.assertEqual(True, x)

    def test_osm_cellids(self):
        init_('../datipisa/segments_osm_cellids_1M_uint64.csv', "../datipisa/osm_cellids_1M_uint64")
        pgm = pgm_index(neuroni, init, slope, intercept, False)
        y = pgm.predict(chiavi).reshape(-1, 1)
        index = [0]
        for i in range(1, len(chiavi)):
            if chiavi[i] == chiavi[i - 1]:
                index.append(index[i - 1])
            else:
                index.append(i)
        err = []

        for i in range(len(chiavi)):
            diff = abs(y[i, 0] - index[i])
            err.append(diff)

        x = all(i <= 64.1 for i in err)
        self.assertEqual(True, x)


if __name__ == '__main__':
    unittest.main()
