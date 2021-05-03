import stumpy
import numpy as np
from numba import cuda


import pandas as pd

 
import os
import os.path
import csv

numbers = []
all_gpu_devices = [device.id for device in cuda.list_devices()] 
#Open the file
with open('243_UCR_Anomaly_100000.txt') as fp:
    #Iterate through each line
    for line in fp:

        numbers.extend( #Append the list of numbers to the result array
            [float(item) #Convert each number to an integer
             for item in line.split() #Split each line of whitespace
             ])
        
window_size=100
name = 243
th = 100000
    
matrix_profile = stumpy.gpu_stump(numbers, m=window_size, device_id=all_gpu_devices)
    
sort = np.argsort(-matrix_profile[:, 0])
    
    
    
discord_idx = sort[np.argmax(sort > th)]
    
if discord_idx < th:
    discord_idx = th
    
print([name,discord_idx])