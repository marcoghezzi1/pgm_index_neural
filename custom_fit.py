import tensorflow as tf
from tensorflow import keras
from modello import pgm_index
from sys import maxsize
import numpy as np


# --- test ---
'''
y1 = 5x+3
y2 = 3x-4
y3 = 10x+7
y4 = 27x-10
y5 = 2x+13
y6 = 9x-14
'''

w = np.array([1, 8, 20, 25, 34, 39]).reshape(1, 6)
slope = np.array([5, 3, 10, 27, 2, 9]).reshape(1, 6)
intercept = np.array([3, -4, 7, -10, 13, -14]).reshape(1, 6)
neuroni = 6

pgm = pgm_index(neuroni, w, slope, intercept, True)
x = np.array([7, 10, 10, 27, 38])
y = np.array([0, 1, 1, 3, 4])
# print(pgm.summary())
pgm.compile(loss='mae', optimizer='adam')
pgm.fit(x, y, epochs=100, batch_size=1)
print(pgm.get_weights())




