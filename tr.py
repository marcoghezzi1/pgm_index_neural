import os
import sys
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from modello import pgm_index
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

dataset = 'datipisa/segments_wiki_ts_1M_uint64.csv'
with open(dataset) as file:
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
neuroni = len(indice)
pgm = pgm_index(neuroni, init, slope, intercept, False)

# calcolo media con valori presi dai dati
y = pgm.predict(x_train)

err = []
for i in range(len(x_train)):
    diff = abs(y[i, 0] - index[i])
    err.append(diff)
err_max_init = np.amax(err)
err_medio_init = np.average(err)

# training con slope inizializzate da 0 a 1e-3 e intercette da -2 a 2
pgm = pgm_index(neuroni, init, slope, intercept, True)

lr = float(sys.argv[1])
opt = Nadam(learning_rate=lr)
opt_name = 'Nadam'
loss_name = 'mean_absolute_error'
pgm.compile(loss=loss_name, optimizer=opt)
y_train = np.array(index).reshape(len(x_train), 1)
batch = int(sys.argv[2])
epoche = int(sys.argv[3])
mc = ModelCheckpoint("Best_PGM_model"+str(lr)+str(batch), monitor='loss', mode='min',
                     save_best_only=True, save_weights_only=True)
es = EarlyStopping(monitor='loss')
history = pgm.fit(x_train, y_train, batch_size=batch, epochs=epoche, verbose=1, callbacks=[mc])
pgm.load_weights("Best_PGM_model"+str(lr)+str(batch))



y = pgm.predict(x_train)
err = []
for i in range(len(x_train)):
    diff = abs(y[i, 0] - index[i])
    err.append(diff)
err_max = np.amax(err)
err_medio = np.average(err)

# update tabella risultati

df = pd.read_csv(r'results_training.csv')
df = df.append({'dataset': dataset, 'err medio iniziale': err_medio_init, 'loss': loss_name,
                'optimizer': opt_name, 'lr': lr, 'batch size': batch,
                'epochs': epoche, 'err medio finale': err_medio, 'errore massimo': err_max}, ignore_index=True)
df.to_csv('results_training.csv', index=False)

# grafico della loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='MAE')
plt.title('MAE')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
log_img = 'logs/plot/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.eps'
plt.savefig(log_img, format='eps')

