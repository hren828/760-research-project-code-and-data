# -*- coding: utf-8 -*-
"""
Created on Mon May  3 00:47:50 2021

@author: ren
"""

import stumpy
import numpy as np
from numba import cuda


import pandas as pd

 
import os
import os.path
import csv

rootdir = "D:/textbook/2021.03/760/23/"


#out = open('sub.csv','a', newline='')
#csv_write = csv.writer(out,dialect='excel')

filenamelist = list()
all_gpu_devices = [device.id for device in cuda.list_devices()] 

 
for parent,dirnames,filenames in os.walk(rootdir):
    for filename in filenames:
        if ".txt" in filename:
            filenamelist.append(filename)
    

            
print("over 1")
            
for index, value in enumerate(filenamelist):
    df=pd.read_csv(rootdir+value, names=['values'])

    
    window_size=100
    name = int(value[:3])
    th = int(value[:-4].split('_')[-1])
    
    matrix_profile = stumpy.gpu_stump(df['values'].values, m=window_size, device_id=all_gpu_devices)
    
    sort = np.argsort(-matrix_profile[:, 0])
    
    
    
    discord_idx = sort[np.argmax(sort > th)]
    
    if discord_idx < th:
        discord_idx = th
    
    print([name,discord_idx])
    
        
    #csv_write.writerow([name ,discord_idx])
    
#out.close()









