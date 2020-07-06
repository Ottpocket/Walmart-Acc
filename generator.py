# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 06:02:58 2020

@author: andre
"""
import numpy as np
import pandas as pd
import os 
import gc

import tensorflow as tf
os.chdir('C:/Users/andre/Desktop/Kaggle/M5 Data')


#main ideas from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#additional info from https://www.machinecurve.com/index.php/2020/04/06/using-simple-generators-to-flow-data-from-file-with-keras/
###############################################################################
#DataGenerator: the generator for a neural network with two input branches and
# one outout.  Use for a nn where LHS (left branch) uses different features than
# RHS (right branch).  
#INPUT:
    #list_IDs: (list) IDs of the samples to be used
    #batch_size: (int) size of minibatch
    #features_LHS: (int) the number of channels for the LHS
    #features_RHS: (int) the number of channels for the RHS
    #window: (int) how much the examples slide each time
    #lag: (int) the period of time the LHS of the network evaluates.
    #eval_: (int) the period of time the RHS of the network evaluates.
    #shuffle: (boolean) does the network shuffle at the start of each epoch?
#OUTPUT:
    #LHS: (3dndarray (batch_size, lag, features_LHS)) minibatch fro LHS
    #RHS: (3dndarray (batch_size, eval, features_RHS))
#GLOBAL VARS:
        #id_dict: (dict of dicts) 
            #item: index of item, 
            #index: day to start the LHS of the network.  Used to calculate 
                #the LHS, RHS, and y.
        #sales: (2dndarray (item_id, some_long_number)) sales data 
            #only for all items.  from to sales_train_eval.csv.
        #sales_features: (3dndarray (item_id, some_long_number, features_RHS) 
            #Used for the 28 days following the final LHS day
###############################################################################
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, features_LHS = 1, features_RHS= 2, #window=14,
                 lag=365, eval_ = 28, shuffle=True):
        'Initialization'
        self.dim1 = (None, lag, 1) #dim of 365 day warm-up
        self.dim2 = (None, eval_, features) #dim of 28 day features 
        self.batch_size = batch_size
        self.list_IDs = list_IDs #The ids to be trained on here
        self.features_LHS = features_LHS
        self.features_RHS = features_RHS
        #self.window = window #window used outside
        self.lag = lag
        self.eval_ = eval_
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        #Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        LHS, RHS, y = self.__data_generation(list_IDs_temp)

        return (LHS, RHS), y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))#indexes of the ids used 
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        LHS = np.empty((self.batch_size, self.dim1[1], self.features_LHS)) #(None, lag) 
        RHS = np.empty((self.batch_size, self.dim2[1], self.features_RHS))#(None,eval, features)
        y = np.empty((self.batch_size, self.dim2[1]))#(None, eval)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):            
            item = id_dict[ID]['item']
            LHS_index = id_dict[ID]['index']
            RHS_index = id_dict[ID]['index'] + self.lag#Same as y index
            
            LHS[i] = sales_3d[item, LHS_index:(LHS_index + self.lag),:]
            RHS[i] = sales_features[item, RHS_index:(RHS_index + self.eval_),:]
            y[i] =   np.reshape(sales_3d[item, RHS_index:(RHS_index + self.eval_)], (self.eval_,))
            
        return LHS, RHS, y
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#Small scale test of the generator.  Appears to work!!!!!
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#Trying it with a Real Data.  Fingers crossed
df = pd.read_csv('C:/Users/andre/Desktop/Kaggle/M5 Data/sales_train_evaluation.csv.zip')
sales = df[df.columns[6:]].astype(np.int16).values
sales = sales[0:500]#Just to make the simulation faster.  No other reason
sales_3d = np.reshape(sales, newshape=(sales.shape[0],sales.shape[1],1))
del df
gc.collect()
features=3
sales_features = np.zeros(shape = (sales.shape[0], sales.shape[1], features))
for i in range(sales_features.shape[0]):    
    for j in range(sales_features.shape[1]):
        start = sales[i,j]
        sales_features[i,j] = np.arange(start= start, stop= start+features)
    if i % 100 == 0:
        print('Finished {} examples'.format(i))

#Creating the id_dict
id_dict = {}
key = 0
window = 14
for item in range(sales.shape[0]):
    for index in range(111):
        id_dict[key] = {'item':item, 'index': index * window}
        key = key +1
        
lag=365
eval_=28

#Testing to see if this can train a network well.  Fingers crossed...
os.chdir('C:/Users/andre/Documents/GitHub/Walmart-Acc/')
from Model import small_model
model = small_model(lhs_features = 1, rhs_features = features)
print(model.summary())


# Parameters
params = {'batch_size': 16, 
          'features_RHS': features, 
          'lag': lag,
          'eval_': eval_, 
          'shuffle': True}

# Generators
training_generator = DataGenerator(list_IDs = [i for i in range(40000)], **params)
validation_generator = DataGenerator(list_IDs = [i for i in np.arange(start=40000, stop=55500)], 
                                                 **params)
model.fit(x = training_generator, epochs=5, validation_data=validation_generator)
model.summary()



#Looking at outputs from the generator
cow = DataGenerator(list_IDs=[i for i in range(55500)], batch_size=16, features_RHS=features, lag=lag,
                    eval_=eval_, shuffle=True)
cow.__len__()#a
l, r, y = cow._DataGenerator__data_generation([109,110]) #Appears to work!


