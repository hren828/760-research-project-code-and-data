import numpy as np
import time
from sklearn import preprocessing
import os.path
import re
import pandas as pd
import random

from autoencoder import AutoEncoder
from lstm_ad import LSTMAD
from lstm_enc_dec_axl import LSTMED
from rnn_ebm import RecurrentEBM

time_start = time.time()
rootdir = "E://760//Project//PhaseI//"

filenamelist = list()

for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        if ".txt" in filename:
            filenamelist.append(filename)
accuracy_list = []

for j in range(20):
    result_list = []
    anomaly_list = []
    a = 0
    for index, value in enumerate(filenamelist):
        df = pd.read_csv(rootdir + value, names=['values'])
        data = df.values
        # preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        data = min_max_scaler.fit_transform(data)

        split_point = int(re.findall("\d+", value)[1])
        data = data[0:split_point, ]

        # Get the split point of training and test set
        split_point = int(np.floor(0.7 * split_point))

        # Add anomaly
        mu = 0
        # sigma = 0.4, 0.6, 0.8
        sigma = 0.4
        anomaly_point = random.randint(split_point, data.shape[0]-1)
        anomaly_list.append(anomaly_point)
        temp = random.gauss(mu, sigma)
        if 0 <= temp < 0.3 or -0.3 < temp <= 0:
            temp = random.gauss(mu, sigma)
        data[anomaly_point] += temp

        '''
        # Add noise
        # 0.5% noise
        mu = 0
        sigma = 0.3
        noise_amount = int(np.ceil(0.005 * data.shape[0]))
        if noise_amount < 5:
            noise_amount = 5

        for i in range(noise_amount):
            noise_point = random.randint(0, data.shape[0]-2)
            data[noise_point] += random.gauss(mu, sigma)
        '''


        # 2% noise
        mu = 0
        sigma = 0.3
        noise_amount = np.ceil(0.02 * data.shape[0])
        if noise_amount < 20: 
            noise_amount = 20

        for ll in range(int(noise_amount)):
            noise_point = random.randint(0, data.shape[0]-1)
            data[noise_point] += random.gauss(mu, sigma)



        # Resplit data
        train = data[0:split_point, ]
        train = pd.DataFrame(train)
        test = data[split_point:data.shape[0], ]
        test = pd.DataFrame(test)

        '''
        # LSTMAD model
        model = LSTMAD()
        model.fit(train)

        # LSTMED model
        model = LSTMED(hidden_size=10)
        model.fit(train)

        # REBM model
        model = RecurrentEBM(hidden_size=10)
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

    right = 0
    for i in range(len(result_list)):
        if i != 24:
            if result_list[i] in range(anomaly_list[i] - 101, anomaly_list[i] + 101):
                right += 1
        else:
            if result_list[i] in range(anomaly_list[i] - 11, anomaly_list[i] + 11):
                right += 1

    print("The accuracy is: ", right / 25)
    accuracy_list.append(right / 25)

print("The mean accuracy of 20 times is: ", np.mean(accuracy_list))
print("The accuracy variance of 20 times is: ", np.var(accuracy_list))

time_end = time.time()
print("running time is:", time_end - time_start)
