# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 03:49:25 2020

@author: andre
"""

import tensorflow as tf
from tensorflow.keras.layers import *
from time import time
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses, models, optimizers

###############################################################################
#Small model
###############################################################################

def small_model(lhs_features, rhs_features):
    #Branch 1
    inp1 = Input(shape=(365,lhs_features))
    x = MaxPooling1D(pool_size=13)(inp1)
    x = Conv1D(filters = 5, kernel_size = 5, padding = 'same')(x)
    
    #Branch 2
    inp2 = Input(shape=(28, rhs_features))
    y = Conv1D(filters=3, kernel_size=1, padding='same')(inp2)
    
    z = tf.concat([x,y], axis=2)
    z = Dense(1, activation='relu')(z)
    out = Reshape(target_shape=(28,))(z)
    model = models.Model(inputs = [inp1, inp2], outputs = out)
    
    opt = Adam(lr = .01)
    losses_ = tf.keras.losses.MSE
    model.compile(loss = losses_, optimizer = opt, metrics = ['mean_absolute_error'])
    return model

'''
###############################################################################
#Competition model
###############################################################################
inp1 = Input(shape=(365,1))
x = MaxPooling1D(pool_size=13)(inp1)
x = Conv1D(filters = 5, kernel_size = 5, padding = 'same')(x)

#Branch 2
features = 10
inp2 = Input(shape=(28,features))
y = Conv1D(filters=3, kernel_size=1, padding='same')(inp2)

z = tf.concat([x,y], axis=2)
z = Dense(1, activation='relu')(z)
out = Reshape(target_shape=(28,))(z)
model = models.Model(inputs = [inp1, inp2], outputs = out)
model.summary()
'''