import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matrixprofile as mp

from matrixprofile import *

import string
 
import os
import os.path
import csv

rootdir = "D:/textbook/2021.03/760/data_phase2/"


out = open('sub.csv','a', newline='')
csv_write = csv.writer(out,dialect='excel')

filenamelist = list()


 
for parent,dirnames,filenames in os.walk(rootdir):
    for filename in filenames:
        if ".txt" in filename:
            filenamelist.append(filename)
    
print("over 1")
            
for index, value in enumerate(filenamelist):
    df=pd.read_csv(rootdir+value, names=['values'])
    #plt.figure(figsize=(20,5)) 
    #plt.plot(df.index,df.values)
    #plt.axvline(x=int(value[:-4].split('_')[-1]), c="r", ls="--", lw=2)
    #plt.title(index)
    #plt.show()
    
    window_size=100

    profile=mp.compute(df['values'].values, window_size)

    profile = mp.discover.discords(profile,k=1)['discords']    
    
    print(index+1)
    
    
    
    csv_write.writerow([index+1 ,profile[0]])
    
out.close()
