import os.path
import pandas as pd
import matplotlib.pyplot as plt
import matrixprofile as mp
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
import re
from sklearn import preprocessing
from statsmodels.tsa.stattools import adfuller
import random





# LOF noise experiment
rootdir = "E://760//Project//PhaseI//"

filenamelist = list()

for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        if ".txt" in filename:
            filenamelist.append(filename)

accuracy_list = []
for k in range(10):
    result_list = []
    anomaly_list = []
    for index, value in enumerate(filenamelist):
        df = pd.read_csv(rootdir + value, names=['values'])

        split_point = int(re.findall("\d+", value)[1])

        data = df.values[0:(split_point), ]
        min_max_scaler = preprocessing.MinMaxScaler()
        data = min_max_scaler.fit_transform(data)

        # Get the split point of training and test set
        split_point = int(np.floor(0.7 * split_point))

        # Add anomaly
        mu = 0
        sigma = 0.4
        anomaly_point = random.randint(split_point, data.shape[0] - 1)
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

        if noise_amount <= 5:
            noise_amount = 5

        for i in range(noise_amount):
            noise_point = random.randint(0, data.shape[0] - 1)
            data[noise_point] += random.gauss(mu, sigma)
        '''


        # 2% noise
        mu = 0
        sigma = 0.3
        noise_amount = int(np.ceil(0.02 * data.shape[0]))

        if noise_amount <= 15:
            noise_amount = 15

        for i in range(noise_amount):
            noise_point = random.randint(0, data.shape[0]-1)
            data[noise_point] += random.gauss(mu, sigma)


        outlier_fraction = 50 / df.shape[0]
        clf = LocalOutlierFactor(n_neighbors=50, contamination=outlier_fraction)
        y = clf.fit_predict(data)
        score = -clf.negative_outlier_factor_
        result = np.argsort(score)[::-1]
        result_list.append(result[0])


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



'''
# Isolation forest noise experiment
rootdir = "E://760//Project//PhaseI//"

filenamelist = list()

for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        if ".txt" in filename:
            filenamelist.append(filename)

accuracy_list = []
for k in range(20):
    result_list = []
    anomaly_list = []
    for index, value in enumerate(filenamelist):
        df = pd.read_csv(rootdir + value, names=['values'])

        split_point = int(re.findall("\d+", value)[1])

        data = df.values[0:(split_point), ]
        min_max_scaler = preprocessing.MinMaxScaler()
        data = min_max_scaler.fit_transform(data)

        # Get the split point of training and test set
        split_point = int(np.floor(0.7 * split_point))

        # Add anomaly
        mu = 0
        # sigma = 0.4, 0.6, 0.8
        sigma = 0.4
        anomaly_point = random.randint(split_point, data.shape[0] - 1)
        anomaly_list.append(anomaly_point)
        temp = random.gauss(mu, sigma)
        if 0 <= temp < 0.3 or -0.3 < temp <= 0:
            temp = random.gauss(mu, sigma)
        data[anomaly_point] += temp

        # Add noise
        
        # 0.5% noise
        mu = 0
        sigma = 0.3
        noise_amount = int(np.ceil(0.005 * data.shape[0]))

        if noise_amount <= 5:
            noise_amount = 5

        for i in range(noise_amount):
            noise_point = random.randint(0, data.shape[0] - 1)
            data[noise_point] += random.gauss(mu, sigma)
        

        
        # 2% noise
        mu = 0
        sigma = 0.3
        noise_amount = int(np.ceil(0.02 * data.shape[0]))

        if noise_amount <= 15:
            noise_amount = 15

        for i in range(noise_amount):
            noise_point = random.randint(0, data.shape[0] - 1)
            data[noise_point] += random.gauss(mu, sigma)
        

        outlier_fraction = 50 / df.shape[0]
        clf = IsolationForest(n_estimators=200, contamination=outlier_fraction)
        y = clf.fit_predict(data)
        score = -clf.score_samples(data)
        result = np.argsort(score)[::-1]
        result_list.append(result[0])

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
'''