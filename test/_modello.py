import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from modello import pgm_index


with open('../datipisa/segments_wiki_ts_1M_uint64.csv') as file:
    indice = pd.read_csv(file, header=2)
print(indice[:10])
dati = np.fromfile("../datipisa/wiki_ts_1M_uint64", dtype=np.uint64)

dati = dati.reshape(len(dati), 1)
#print(dati)
init = indice['key']
slope = tf.convert_to_tensor(indice['slope'])
intercept = tf.convert_to_tensor(indice['intercept'])
print(slope)
end = indice['key'][1:].to_numpy()
end = np.append(end, sys.maxsize)
neuroni = len(indice)
pgm = pgm_index(neuroni, init, end, slope, intercept, False)
y = pgm.predict(dati).reshape(-1, 1)
print(type(y))
index = [0]
for i in range(1, len(dati)):
    if dati[i] == dati[i - 1]:
        index.append(index[i-1])
    else:
        index.append(i)
err = []

for i in range(len(dati)):
    diff = abs(y[i, 0]-index[i])
    err.append(diff)

print(all(i <= 64.1 for i in err))
print(len(err))
with open('out.txt', 'w') as f:
    print(err, file=f)






