from sklearn import svm
import numpy as np
import math
from sklearn.model_selection import KFold
import e2LSH
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
import os.path
import re
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from autoencoder import AutoEncoder
from lstm_ad import LSTMAD
from lstm_enc_dec_axl import LSTMED
from dagmm import DAGMM
from rnn_ebm import RecurrentEBM

time_start = time.time()
rootdir = "E://760//Project//PhaseII//"

filenamelist = list()

for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        if ".txt" in filename:
            filenamelist.append(filename)

result_list = []
a = 0
for index, value in enumerate(filenamelist):
    if index in [203, 204, 205, 206, 207, 224, 225, 238, 239, 240, 241, 242, 243, 249, 250]:
        result_list.append(0)
        a += 1
        print(a)
        continue
    df = pd.read_csv(rootdir + value, names=['values'])
    data = df.values
    # preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    split_point = int(re.findall("\d+", value)[1])

    train = data[0:split_point, ]
    train = pd.DataFrame(train)
    test = data[split_point:data.shape[0], ]
    test = pd.DataFrame(test)

    '''
    #LSTMAD
    model = LSTMAD()
    model.fit(train)

    #LSTMED
    model = LSTMED(hidden_size=10)
    model.fit(train)

    model = LSTMED(hidden_size=10)
    model.fit(train)

    #DAGMM model
    model = DAGMM()
    model.fit(train)
    '''

    model = AutoEncoder(num_epochs=40, hidden_size=10)
    model.fit(train)

    error = model.predict(test)
    error = np.abs(error)

    result = np.argmax(error) + split_point
    result_list.append(result)

    a += 1
    print(a)

df_output = pd.DataFrame({'No.': range(1, len(filenamelist) + 1), 'location': result_list})
df_output.to_csv("Output.csv", index=False, sep=',')

time_end = time.time()
print("running time is:", time_end - time_start)