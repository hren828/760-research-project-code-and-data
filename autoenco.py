import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from matplotlib import pyplot as plt
import csv
import os
import os.path


#filename = '015_UCR_Anomaly_5000.txt'
rootdir = "/home/mist/dp/"

out = open('test.csv','a', newline='')
csv_write = csv.writer(out,dialect='excel')

filenamelist = list()
for parent,dirnames,filenames in os.walk(rootdir):
    for filename in filenames:
        if ".txt" in filename:
            filenamelist.append(filename)
            
print("over 1")


#data = pd.read_csv(filename,names=['value'])
#point = 5000
n_steps = 100

def create_sequences(values, steps=n_steps):
    output = []
    for i in range(len(values) - steps):
        output.append(values[i : (i + steps)])
    return np.stack(output)

def normalize_test(values, mean, std):
    values -= mean
    values /= std
    return values

for index, value in enumerate(filenamelist):
    name = int(value[:3])
    data = pd.read_csv(rootdir+value,names=['value'])
    point = int(value[:-4].split('_')[-1])

    train = data.iloc[:point].reset_index(drop=True)
    test  = data.iloc[point:].reset_index(drop=True)

    training_mean = train.mean()
    training_std = train.std()
    training_value = (train - training_mean) / training_std

    x_train = create_sequences(training_value.values)
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(
                filters=16, kernel_size=4, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=8, kernel_size=4, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=8, kernel_size=4, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=16, kernel_size=4, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mse")
    dota_model = model.fit(
                        x_train,
                        x_train,
                        epochs=2009,
                        batch_size=64,
                        validation_split=0.2,
                        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")],
                        verbose=0
                    )
    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
    threshold = np.max(train_mae_loss)
    df_test_value = (test - training_mean) / training_std
    x_test = create_sequences(df_test_value.values)
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
    test_mae_loss = test_mae_loss.reshape((-1))
    ano = test_mae_loss - threshold

    idex = np.where(ano == max(ano))[0][0]+point
    
    print([name,idex])
    
    csv_write.writerow([name ,idex])
    
out.close()