# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:13:29 2020

@author: andre
"""

import pandas as pd
import numpy as np
import os 
import gc
from time import time
os.chdir('C:/Users/andre/Desktop/Kaggle/M5 Data')
###############################################################################
#Rough NN_Trend data: this is 
###############################################################################
#Downloading the data
calendar_df = pd.read_csv('calendar.csv')
stv_df = pd.read_csv('sales_train_validation.csv.zip')


#Making a tidy df
id_cols = ['id','item_id','dept_id','cat_id','store_id','state_id']
TARGET = 'sales'
df = pd.melt(stv_df, id_vars= id_cols, var_name = 'd', value_name = TARGET)

#memory reduction
for col in df.columns:
    if df[col].dtype == 'object':
        start = time()
        print(f'Converting {col} from object to category')
        df[col] = df[col].astype('category')
        print(f'Took {time() - start} seconds.')



#Horrific FE/Memory reduction
calendar_features = ['snap_CA', 'snap_WI', 'month','year']
for col in calendar_features:
    calendar_df[col] = calendar_df[col].astype('int16')

df = df.merge(calendar_df[['snap_CA', 'snap_WI', 'month','year','d']],
                 how='left', on='d')
df['d'] = df['d'].str[2:].astype('int16')
#Turning the data to 1D-Conv readable data
tiny = df[(df['id'] == 'HOBBIES_1_001_CA_1_validation') | (df['id'] == 'HOBBIES_1_002_CA_1_validation')]

#DO NOT DO THE FOLLOWING
#tiny.groupby('id').apply(lambda df_: df_['year'].max())
#because 'id' is categorical, the groupby will preform the apply lambdas on
# all categories, even those not included.  
tiny.dtypes
tiny.id
mapping = {'HOBBIES_1_001_CA_1_validation':1,
           'HOBBIES_1_002_CA_1_validation':2}
tiny['id_map'] = tiny['id'].map(mapping).astype('int8')
features = ['sales','snap_CA','d']
cow = np.array(list(tiny.groupby('id_map').apply(lambda df: df[features].values)))

cow[0,1910:,:]
cow[1,1910:,:] #IT WORKS! DO IT THIS WAY!!!!
del tiny, cow


###############################################################################
#What we want are train data of the form:
    #(np.array(365 days, features), nparray(28 days, features))
#The 28 days will be the first 28 days of the next month after the 365 days.
#Several Ways to do this. 

#df['yr_mon'] = df['year'].astype('str') + '_' +df['month'].astype('str')

for year in [2011, 2012, 2013, 2014, 2015,2016]:
    for month in range(1,13):
        df[(df['year']==2012) & (df['month']==2)].d.min()

cow['year'].astype('str') + '_' +cow['month'].astype('str')



import multiprocessing
multiprocessing.cpu_count()