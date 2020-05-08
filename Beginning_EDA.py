# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:27:55 2020
taken from: https://www.kaggle.com/ryuheeeei/let-s-start-from-here-beginners-data-analysis
@author: andre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import gc
#import lightgbm as lgb
import time
from sklearn.metrics import r2_score
import os 
os.chdir('C:/Users/andre/Documents/Github/Walmart-ACC')

#from https://stackoverflow.com/questions/44575251/reading-multiple-files-contained-in-a-zip-file-with-pandas
from zipfile import ZipFile
zip_file = ZipFile('m5-forecasting-accuracy.zip')
calendar_df = pd.read_csv(zip_file.open('calendar.csv'))
sell_prices_df = pd.read_csv(zip_file.open('sell_prices.csv'))
sales_train_validation_df = pd.read_csv(zip_file.open('sales_train_validation.csv'))
sample_submission_df = pd.read_csv(zip_file.open('sample_submission.csv'))

#sell_prices_df.info(memory_usage='deep') #957.5MB this is around 3x bigger than Mnist
#calendar_df.info(memory_usage='deep')
#sales_train_validation_df.info(memory_usage = 'deep')

###############################################################################
#Memory Reduction
###############################################################################
# Calendar data type cast -> Memory Usage Reduction
calendar_df[["month", "snap_CA", "snap_TX", "snap_WI", "wday"]] = calendar_df[["month", "snap_CA", "snap_TX", "snap_WI", "wday"]].astype("int8")
calendar_df[["wm_yr_wk", "year"]] = calendar_df[["wm_yr_wk", "year"]].astype("int16") 
calendar_df["date"] = calendar_df["date"].astype("datetime64")

nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
for feature in nan_features:
    calendar_df[feature].fillna('unknown', inplace = True)
calendar_df[["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]] = calendar_df[["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]] .astype("category")
calendar_df.info(memory_usage='deep')# 167.8kb.  Huge Memory Reduction!

# Sales Training dataset cast -> Memory Usage Reduction
sales_train_validation_df.loc[:, "d_1":] = sales_train_validation_df.loc[:, "d_1":].astype("int16") #123MB
# Make ID column to sell_price dataframe
sell_prices_df.loc[:, "id"] = sell_prices_df.loc[:, "item_id"] + "_" + sell_prices_df.loc[:, "store_id"] + "_validation"
sell_prices_df = pd.concat([sell_prices_df, sell_prices_df["item_id"].str.split("_", expand=True)], axis=1)
sell_prices_df = sell_prices_df.rename(columns={0:"cat_id", 1:"dept_id"})
sell_prices_df[["store_id", "item_id", "cat_id", "dept_id"]] = sell_prices_df[["store_id","item_id", "cat_id", "dept_id"]].astype("category")
sell_prices_df = sell_prices_df.drop(columns=2)
#sell_prices_df.info(memory_usage='deep') # 696.8MB  About a 30% reduction in size!


###############################################################################
#Creating big DataFrame
###############################################################################
def make_dataframe():
    # Wide format dataset 
    df_wide_train = sales_train_validation_df.drop(columns=["item_id", "dept_id", "cat_id", "state_id","store_id", "id"]).T
    df_wide_train.index = calendar_df["date"][:1913]
    df_wide_train.columns = sales_train_validation_df["id"]
    
    # Making test label dataset
    df_wide_test = pd.DataFrame(np.zeros(shape=(56, len(df_wide_train.columns))), index=calendar_df.date[1913:], columns=df_wide_train.columns)
    df_wide = pd.concat([df_wide_train, df_wide_test])

    # Convert wide format to long format
    df_long = df_wide.stack().reset_index(1)
    df_long.columns = ["id", "value"]

    del df_wide_train, df_wide_test, df_wide
    gc.collect()
    
    df = pd.merge(pd.merge(df_long.reset_index(), calendar_df, on="date"), sell_prices_df, on=["id", "wm_yr_wk"])#6.1GB!
    #df = df.drop(columns=["d"])
    #     df[["cat_id", "store_id", "item_id", "id", "dept_id"]] = df[["cat_id"", store_id", "item_id", "id", "dept_id"]].astype("category")
    df["sell_price"] = df["sell_price"].astype("float16")   
    df["value"] = df["value"].astype("int32")
    df["state_id"] = df["store_id"].str[:2].astype("category")


    del df_long
    gc.collect()
    return df

df = make_dataframe()

#df.info(memory_usage = 'deep')#5.7GB!  Huge
df.info()
df.dtypes
def add_date_feature(df):
    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["week"] = df["date"].dt.week.astype("int8")
    df["day"] = df["date"].dt.day.astype("int8")
    df["quarter"]  = df["date"].dt.quarter.astype("int8")
    return df
df = add_date_feature(df)
