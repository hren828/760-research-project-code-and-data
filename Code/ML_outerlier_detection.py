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

'''
#LOF
rootdir = "E://760//Project//PhaseII//"

filenamelist = list()

for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        if ".txt" in filename:
            filenamelist.append(filename)

result_list = []
a = 0
for index, value in enumerate(filenamelist):
    if index in [203,204,205,206,207,224,225,241,242,238,239,240,241,242,243]:
        result_list.append(0)
        a += 1
        print(a)
        continue
    df = pd.read_csv(rootdir + value, names=['values'])
    data = df.values
    #preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    split_point = int(re.findall("\d+", value)[1])

    outlier_fraction = 1 / df.shape[0]
    clf = LocalOutlierFactor(n_neighbors=50, contamination=outlier_fraction)
    y = clf.fit_predict(data)
    score = -clf.negative_outlier_factor_
    result = np.argsort(score)[::-1]
    for i in range(len(result)):
        if result[i] > split_point:
            result_list.append(result[i])
            break
    a += 1
    print(a)



df_output = pd.DataFrame({'No.':range(1, len(filenamelist)+1),'location':result_list})
df_output.to_csv("Output.csv", index=False, sep=',')
'''

'''
#acf and pacf analysis
rootdir = "E://760//Project//PhaseI//"

filenamelist = list()
ADF = []
for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        if ".txt" in filename:
            filenamelist.append(filename)

result_list = []
for index, value in enumerate(filenamelist):
    df = pd.read_csv(rootdir + value, names=['values'])
    plot_acf(df.values)
    plot_pacf(df.values)
    plt.show()
    a = df.values.tolist()
    result = adfuller(df.values.tolist())
    result_list.append(result)
'''


'''
#Isolation Forest
rootdir = "E://760//Project//PhaseI//"

filenamelist = list()

for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        if ".txt" in filename:
            filenamelist.append(filename)

result_list = []
for index, value in enumerate(filenamelist):
    df = pd.read_csv(rootdir + value, names=['values'])

    split_point = int(re.findall("\d+", value)[1])

    outlier_fraction = 1 / split_point
    clf = IsolationForest(random_state=123, contamination=outlier_fraction, behaviour='new')
    clf.fit(df.values)
    score = -clf.score_samples(df.values)
    result = np.argsort(score)[::-1]
    for i in range(len(result)):
        if result[i] > split_point:
            result_list.append(result[i])
            break

df_output = pd.DataFrame({'No.':range(1, len(filenamelist)+1),'location':result_list})
df_output.to_csv("Output.csv", index=False, sep=',')
'''


'''
#LOF add noise training
rootdir = "E://760//Project//PhaseI//"

filenamelist = list()

for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        if ".txt" in filename:
            filenamelist.append(filename)

result_list = []
for index, value in enumerate(filenamelist):
    df = pd.read_csv(rootdir + value, names=['values'])

    split_point = int(re.findall("\d+", value)[1])

    data = df.values[0:(split_point),]
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    # add noise
    mu = 0
    sigma = 1
    data[2000] += random.gauss(mu, sigma)

    outlier_fraction = 50 / df.shape[0]
    clf = LocalOutlierFactor(n_neighbors=50, contamination=outlier_fraction)
    y = clf.fit_predict(data)
    score = -clf.negative_outlier_factor_
    result = np.argsort(score)[::-1]
    result_list.append(result[0])



df_output = pd.DataFrame({'No.':range(1, len(filenamelist)+1),'location':result_list})
df_output.to_csv("Output.csv", index=False, sep=',')
'''


#Isolation Forest add noise training
rootdir = "E://760//Project//PhaseI//"

filenamelist = list()

for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        if ".txt" in filename:
            filenamelist.append(filename)

result_list = []
for index, value in enumerate(filenamelist):
    df = pd.read_csv(rootdir + value, names=['values'])

    split_point = int(re.findall("\d+", value)[1])

    data = df.values[0:(split_point),]
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    # add noise
    mu = 0
    sigma = 1
    data[2000] += random.gauss(mu, sigma)

    plt.figure(figsize=(20,5))
    plt.plot(data)
    plt.axvline(x=2000, c="r", ls="--", lw=2)
    plt.title(index)
    plt.show()

    outlier_fraction = 50 / df.shape[0]
    clf = IsolationForest(random_state=123, contamination=outlier_fraction, behaviour='new')
    clf.fit(data)
    score = -clf.score_samples(data)
    result = np.argsort(score)[::-1]
    result_list.append(result[0])



df_output = pd.DataFrame({'No.':range(1, len(filenamelist)+1),'location':result_list})
df_output.to_csv("Output.csv", index=False, sep=',')

























