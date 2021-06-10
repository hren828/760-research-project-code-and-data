import numpy as np
import time
from sklearn import preprocessing
import os.path
import re
import pandas as pd
import random
from dtw import dtw
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from autoencoder import AutoEncoder
from lstm_ad import LSTMAD
from lstm_enc_dec_axl import LSTMED
from rnn_ebm import RecurrentEBM



rootdir = "E://760//Project//PhaseI//"

filenamelist = list()

for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        if ".txt" in filename:
            filenamelist.append(filename)

result_list = []
anomaly_list = []
data_list = []
split_point_list = []
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
    split_point_list.append(split_point)


    data_list.append(data)

'''
# Get the clusters of 25 datasets
# Use dtw method
distance_matrix = np.zeros((len(data_list),len(data_list)), dtype=float)
a = 0
for i in range(len(data_list)-1):
    for j in range(i+1, len(data_list)):
        if len(data_list[i]) > 20000:
            x = data_list[i][0:20000]
        else:
            x = data_list[i]
        if len(data_list[j]) > 20000:
            y = data_list[j][0:20000]
        else:
            y = data_list[j]

        manhattan_distance = lambda x, y: np.abs(x - y)
        d = dtw(x, y, manhattan_distance)[0]
        distance_matrix[i,j] = d
        distance_matrix[j,i] = d
    a += 1
    print(a)

min_max_scaler = preprocessing.MinMaxScaler()
distance_matrix = min_max_scaler.fit_transform(distance_matrix)

# Use DBSCAN to cluster the distance metric
y_pred = DBSCAN(eps=0.3, min_samples=3, metric='precomputed').fit(distance_matrix)
print()
'''

# List the result labels of clusters
labels = [0,0,0,4,1,1,1,1,1,1,0,0,2,2,0,0,0,2,3,3,3,3,3,4,4]

accuracy_list = []
for kk in range(5):
    result_list = [0] * 25
    for i in range(len(set(labels))):
        index = [k for k, x in enumerate(labels) if x == i]
        data = data_list[index[0]]

        # Add anomaly
        mu = 0
        sigma = 0.8
        anomaly_point = random.randint(split_point_list[index[0]], data.shape[0])
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
            noise_point = random.randint(0, data.shape[0])
            data[noise_point] += random.gauss(mu, sigma)
        '''


        # 2% noise
        mu = 0
        sigma = 0.3
        noise_amount = np.ceil(0.02 * data.shape[0])
        if noise_amount < 20:
            noise_amount = 20

        for ll in range(int(noise_amount)):
            noise_point = random.randint(0, data.shape[0]-2)
            data[noise_point] += random.gauss(mu, sigma)

        # Resplit data
        train = data[0:split_point_list[index[0]], ]
        train = pd.DataFrame(train)
        test = data[split_point_list[index[0]]:data.shape[0], ]
        test = pd.DataFrame(test)

        # Train the DNN models by the basic data
        model = AutoEncoder(num_epochs=40, hidden_size=10)
        model.fit(train)

        error = model.predict(test)
        error = np.abs(error)

        result = np.argmax(error) + split_point_list[index[0]]
        result_list[index[0]] = result

        # Fit the other data in the same clusters
        for j in range(1, len(index)):
            data = data_list[index[j]]

            # Add anomaly
            mu = 0
            sigma = 0.8
            anomaly_point = random.randint(split_point_list[index[j]], data.shape[0] - 1)
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
                noise_point = random.randint(0, data.shape[0])
                data[noise_point] += random.gauss(mu, sigma)
            '''

            '''
            # 2% noise
            mu = 0
            sigma = 0.4
            noise_amount = np.ceil(0.02 * data.shape[0])
            if noise_amount < 20:
                noise_amount = 20

            for i in range(len(noise_amount)):
                noise_point = random.randint(0, data.shape[0])
                data[noise_point] += random.gauss(mu, sigma)
            '''

            # Resplit data
            train = data[0:split_point_list[index[j]], ]
            train = pd.DataFrame(train)
            test = data[split_point_list[index[j]]:data.shape[0], ]
            test = pd.DataFrame(test)

            # Fit the other data in the same
            error = model.predict(test)
            error = np.abs(error)

            result = np.argmax(error) + split_point_list[index[j]]
            result_list[index[j]] = result

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


























