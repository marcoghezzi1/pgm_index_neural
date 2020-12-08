import unittest

import pandas as pd
import numpy as np
import sys
from modello import pgm_index


with open('../datipisa/segments_wiki_ts_1M_uint64.csv') as file:
    indice_wiki = pd.read_csv(file, header=2)
wiki = np.fromfile("../datipisa/wiki_ts_1M_uint64", dtype=np.uint64)

wiki = wiki.reshape(len(wiki), 1)
init_wiki = indice_wiki['key'].to_numpy().reshape(1, len(indice_wiki))
slope_wiki = indice_wiki['slope'].to_numpy().reshape(1, len(indice_wiki))
intercept_wiki = indice_wiki['intercept'].to_numpy().reshape(1, len(indice_wiki))
neuroni_wiki = len(indice_wiki)

with open('../datipisa/segments_books_1M_uint64.csv') as file:
    indice_books = pd.read_csv(file, header=2)
books = np.fromfile("../datipisa/books_1M_uint64", dtype=np.uint64)
books = books.reshape(len(books), 1)
init_books = indice_books['key'].to_numpy().reshape(1, len(indice_books))
slope_books = indice_books['slope'].to_numpy().reshape(1, len(indice_books))
intercept_books = indice_books['intercept'].to_numpy().reshape(1, len(indice_books))
neuroni_books = len(indice_books)


class MyTestCase(unittest.TestCase):
    def test_wiki(self):
        pgm = pgm_index(neuroni_wiki, init_wiki, slope_wiki, intercept_wiki, False)
        y = pgm.predict(wiki).reshape(-1, 1)
        index = [0]
        for i in range(1, len(wiki)):
            if wiki[i] == wiki[i - 1]:
                index.append(index[i - 1])
            else:
                index.append(i)
        err = []

        for i in range(len(wiki)):
            diff = abs(y[i, 0] - index[i])
            err.append(diff)
        x = all(i <= 64.1 for i in err)
        print(np.amax(err))
        self.assertEqual(True, x)

    def test_books(self):
        pgm = pgm_index(neuroni_books, init_books, slope_books, intercept_books, False)
        y = pgm.predict(books).reshape(-1, 1)
        index = [0]
        for i in range(1, len(books)):
            if books[i] == books[i - 1]:
                index.append(index[i - 1])
            else:
                index.append(i)
        err = []

        for i in range(len(books)):
            diff = abs(y[i, 0] - index[i])
            err.append(diff)
        x = all(i <= 64.1 for i in err)
        self.assertEqual(True, x)


if __name__ == '__main__':
    unittest.main()
