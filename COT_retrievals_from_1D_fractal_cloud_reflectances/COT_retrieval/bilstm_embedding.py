'''
    Author: Xiangyang Meng
    Email: xmeng1@umbc.edu
    This code is for BiLSTM with embedding model.
'''


import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.list_physical_devices(device_type=None)
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dropout,MultiHeadAttention,Bidirectional,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate
from tensorflow.keras import callbacks 
from sklearn.model_selection import KFold


os=82
ts=os
ltype = 'mean_squared_error' #evaluation metric MSE 
type2=tf.keras.metrics.RootMeanSquaredError()  #evaluation metric RMSE
bsize = 16 #batch size
eps = 4000 #maximum number of epochs
path_model="saved_model/bilstm_embedding"

#------------------------
# Bi-LSTM embedding model
#------------------------

def eval_model(train_image,train_label,n):
    input_layer = tf.keras.layers.Input(shape=(ts,1),name='Input')
    projection=tf.keras.layers.Dense(64,activation='relu')(input_layer)
    positions = tf.range(start=0, limit=82, delta=1)
    embedding=tf.keras.layers.Embedding(input_dim=(82),output_dim=(64),input_length=128)(positions)
    encoded = projection+embedding

    bi=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, activation='tanh',return_sequences = True))(encoded)
    bi=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, activation='tanh',return_sequences = True))(bi)
    
    flat=tf.keras.layers.Flatten()(bi)

    output=tf.keras.layers.Dense(os, activation="linear")(flat)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam',loss='mse',metrics=[ltype,type2])
    model.summary()

    earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                                mode ="min", patience = 25,  
                                                restore_best_weights = True,
                                               verbose=1) 

    history = model.fit(train_image,train_label,batch_size = bsize,epochs = eps,validation_split=0.125,
                              callbacks =[earlystopping])#validation set(10%)/trainset(80%)=0.125
    model.save(path_model+'/model%d.h5'%(n+1))
    return model,history






