# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 04:20:21 2020

@author: andre
"""

import pandas as pd
import numpy as np
import csv
import os 
os.chdir('C:/Users/andre/Desktop/Kaggle/M5 Data/Post_Online')
df = pd.read_csv('C:/Users/andre/Desktop/Kaggle/M5 Data/sales_train_evaluation.csv.zip')
'''
We are going to transform the 30,490 by 1947 dataframe into a numpy array
with dims slices_per_item by 365
slices_per_item = [1941 - (365+28)] // 14
'''


drop_cols = df.columns[0:6]
df2 = df.drop(drop_cols, axis=1).astype(np.int16).copy()
df2.info()#112.9Mb as np.int16
df2.values.nbytes /1000000000
df2_np = df2.values
np.sum(df2.values>0)
#df2 = df.drop(drop_cols, axis=1).astype('category').copy()
#df2.info()#62.1Mb as Category

#We will start
'''
window = 14
lag = 365
pred = 28
#starts = 
for i in range(150):
    if i*window < (1941 - (lag+pred)):
        in_range = True
    else:
        in_range = False
    print(f'Index {i*14}, iteration {i}: In range: {in_range}')
'''

start_indices = [i*14 for i in range(111)]
items = [0]
with open('fileName.csv', 'wb') as f:
    writer = csv.writer(f, delimiter=',')
    for item in items:
        for index in start_indices:
            writer.writerow(df2_np[item,index:(index+365)])       
        
(1941 - (365+28))//14

df = np.arange(15*15)
df = df.reshape(15,15)

a_file = open("test.csv", "w")
for row in df:
    print(row)
    np.savetxt(a_file, row, fmt= '%i', delimiter = ';', newline='\n')
a_file.close()

for i in range(15):
    np.savetxt('trainingdata2.csv', [df[i]], fmt='%i', newline=" ", delimiter=',')
