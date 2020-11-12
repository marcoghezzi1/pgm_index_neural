import pandas as pd
import numpy as np
import sys
import os
from modello import pgm_index

os.chdir('../datipisa')
with open('segments_wiki_ts_1M_uint64.csv') as file:
    indice = pd.read_csv(file, header=2)

dati = np.fromfile("wiki_ts_1M_uint64", dtype=np.uint64)

dati = dati[:500]
dati = dati.reshape(500, 1).astype("float32")
#print(dati)
init = indice['key'].to_numpy()
slope = indice['slope'].to_numpy()
intercept = indice['intercept'].to_numpy()
end = indice['key'][1:].to_numpy()
end = np.append(end, sys.maxsize)
neuroni = len(indice)
pgm = pgm_index(neuroni, init, end, slope, intercept, False)
y = pgm.predict(dati)
print(y)






