from modello import pgm_index
import pandas as pd
import numpy as np
import sys


with open('datipisa/segments_wiki_ts_1M_uint64.csv') as file:
    indice = pd.read_csv(file, header=2)
x_train = np.fromfile("datipisa/wiki_ts_1M_uint64", dtype=np.uint64)

x_train = x_train.reshape(len(x_train), 1)


index = [0]
for i in range(1, len(x_train)):
    if x_train[i] == x_train[i - 1]:
        index.append(index[i-1])
    else:
        index.append(i)

init = indice['key'].to_numpy().reshape(1, len(indice))
slope = indice['slope'].to_numpy().reshape(1, len(indice))
intercept = indice['intercept'].to_numpy().reshape(1, len(indice))
end = indice['key'][1:]
end = np.append(end, sys.maxsize).reshape(1, len(indice))
neuroni = len(indice)
pgm = pgm_index(neuroni, init, end, slope, intercept, True)


pgm.compile(loss='mean_absolute_error', optimizer='adam')


print(x_train.shape)
y_train = np.array(index).reshape(len(x_train), 1)
print(y_train.shape)


pgm.fit(x_train, y_train)

