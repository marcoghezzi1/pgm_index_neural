import os
import sys
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from modello import pgm_index
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime


# CUSTOM LOSS
def my_loss(y_true, y_pred):
    abs_difference = tf.abs(y_true - y_pred)
    return tf.math.log(K.sum(tf.math.exp(abs_difference)))

dati = True

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
pgm = pgm_index(neuroni, init, slope, intercept, True, dati)

# calcolo media con valori presi dai dati
y = pgm.predict(x_train)

err = []
for i in range(len(x_train)):
    diff = abs(y[i, 0] - index[i])
    err.append(diff)
err_max_init = np.amax(err)
err_medio_init = np.average(err)

init_custom = input("init pesi random? Y/N ")
if init_custom == 'Y':
    dati = False

# training con dati
pgm = pgm_index(neuroni, init, slope, intercept, True, dati)
lr = input("learning rate: ")
lr = float(lr)

batch = input("batch: ")
batch = int(batch)

epoche = input("epoche: ")
epoche = int(epoche)


loss = input("loss (mae/max): ")
if loss == 'max':
    loss = my_loss

opt = Adam(learning_rate=lr)
opt_name = 'Adam'
loss_name = loss
pgm.compile(loss=loss, optimizer=opt)
y_train = np.array(index).reshape(len(x_train), 1).astype(np.float64)
y_train = y_train / len(y_train)
mc = ModelCheckpoint("Best_PGM_model"+str(lr)+str(batch), monitor='loss', mode='min',
                     save_best_only=True, save_weights_only=True)
es = EarlyStopping(monitor='loss')
history = pgm.fit(x_train, y_train, batch_size=batch, epochs=epoche, verbose=1, callbacks=[mc])
pgm.load_weights("Best_PGM_model"+str(lr)+str(batch))



y = pgm.predict(x_train)
y=(y-np.min(y))/(np.max(y)-np.min(y))
y=np.floor(y*len(x_train))

err = []
for i in range(len(x_train)):
    diff = abs(y[i, 0] - index[i])
    err.append(diff)
err_max = np.amax(err)
err_medio = np.average(err)

# update tabella risultati

df = pd.read_csv(r'results.csv')
df = df.append({'dataset': dataset, 'err medio iniziale': err_medio_init, 'loss': loss_name,
                'optimizer': opt_name, 'lr': lr, 'batch size': batch,
                'epochs': epoche, 'err medio finale': err_medio, 'errore massimo': err_max}, ignore_index=True)
df.to_csv('results.csv', index=False)

# grafico della loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='MaxError')
plt.title('MaxError')
plt.ylabel('MaxError value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
log_img = 'logs/plot/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + loss_name + str(lr) + '.eps'
plt.savefig(log_img, format='eps')