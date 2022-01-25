
'''
    Author: Xiangyang Meng
    Email: xmeng1@umbc.edu
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
from tensorflow.keras.layers import MultiHeadAttention,Bidirectional,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate
from tensorflow.keras import callbacks 
from sklearn.model_selection import KFold



os=82
ts=os
ltype = 'mean_squared_error' #evaluation metric MSE 
type2=tf.keras.metrics.RootMeanSquaredError()  #evaluation metric RMSE
bsize = 16 #batch size
eps = 4000 #maximum number of epochs
path_model="saved_model/bilstm_transformer"




#------------------------
# BiLSTM with Transformer
#------------------------

def eval_model(train_image, train_label,n):


    #bi-LSTM
    input_layer = tf.keras.layers.Input(shape=(ts,1),name='Input')
    bi=Bidirectional(tf.keras.layers.LSTM(units=64, activation='tanh',return_sequences = True,input_shape=(ts,1)))(input_layer) 
    bi=Bidirectional(tf.keras.layers.LSTM(units=64, activation='tanh',return_sequences = True))(bi)


    #transformer block 
    slf_attn = MultiHeadAttention(num_heads=3,key_dim=3)(bi,bi) ####understand the key_dim and num_heads
    layernorm1=tf.keras.layers.LayerNormalization(epsilon=1e-6)(slf_attn+bi)
    dense1=tf.keras.layers.Dense(128,activation='relu')(layernorm1)
    layernorm2=tf.keras.layers.LayerNormalization(epsilon=1e-6)(layernorm1+dense1)
    
    flat=tf.keras.layers.Flatten()(layernorm2)
    
    #output
    drop=tf.keras.layers.Dropout(0.2)(flat)
    output = tf.keras.layers.Dense(os,activation='relu')(drop)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam',loss=['mse'],metrics=[ltype,type2])
    model.summary()

    earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                                mode ="min", patience = 25,  
                                                restore_best_weights = True,
                                               verbose=1) 
    history = model.fit(train_image,train_label,batch_size = bsize,epochs = eps,validation_split=0.125,
                              callbacks =[earlystopping])

    model.save(path_model+'/model(%.f).h5'%(n+1))
    return model,history
    
