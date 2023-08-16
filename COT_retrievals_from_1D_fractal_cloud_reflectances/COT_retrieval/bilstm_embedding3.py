'''
    Author: Xiangyang Meng
    Email: xmeng1@umbc.edu
    This code is for BiLSTM with Embedding model to be used with Dataset 3 and Dataset 4.
'''
import tensorflow as tf
import psutil
# physical_devices = tf.config.list_physical_devices('GPU')[3] 
# tf.config.experimental.set_memory_growth(physical_devices, True)
# tf.config.list_physical_devices(device_type=None)
# import tensorflow.keras
# from tensorflow.keras import models
# from tensorflow.keras import layers
# import matplotlib.pyplot as plt
# import numpy as np
# import h5py
from sklearn.utils import shuffle
# from keras.models import Sequential
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras import callbacks 
from keras.constraints import maxnorm
# import keras
# from sklearn.model_selection import KFold



# Set CUDA device explicitly
gpu = tf.config.experimental.list_physical_devices('GPU')[1] 
# print(gpu)
tf.config.experimental.set_visible_devices(gpu, 'GPU')
logical_gpus = tf.config.experimental.list_logical_devices('GPU') 
# print(logical_gpus[0])
# print(tf.config.get_visible_devices('GPU'))

# Verify correct GPU is selected
assert len(logical_gpus) == 1


oss=82
ts=oss
# ltype = 'mean_squared_error' #evaluation metric MSE 
# type2=tf.keras.metrics.RootMeanSquaredError()  #evaluation metric RMSE
bsize = 16 #batch size
eps = 4000 #maximum number of epochs
path_model="saved_model/lstm3"





# Define a custom callback to monitor memory usage after each epoch
class MemoryUsageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Memory used after Epoch {epoch + 1}: {memory_mb:.2f} MB")





#------------
# BiLSTM with Embedding model
#------------



def eval_model(train_image, test_image,train_label_cot,train_label_cer, test_label_cot, test_label_cer,n):



    ltype=tf.keras.metrics.MeanAbsolutePercentageError()
    ltype2 = 'mean_squared_error'
    ltype3=tf.keras.metrics.RootMeanSquaredError()

    bsize = 256
    eps = 4000


     #bi-LSTM
    input_layer = tf.keras.layers.Input(shape=(82,1),name='Input')
    projection=tf.keras.layers.Dense(96,activation='relu')(input_layer)
    positions = tf.range(start=0, limit=82, delta=1)
    embedding=tf.keras.layers.Embedding(input_dim=(96),output_dim=(96),input_length=32)(positions)
        # adds a learnable position embedding to the projected vector
    encoded = projection+embedding

    bi=Bidirectional(tf.keras.layers.LSTM(units=64, activation='tanh',return_sequences = True))(encoded) 
    bi=Bidirectional(tf.keras.layers.LSTM(units=64, activation='tanh',return_sequences = True))(bi)



    #dense for CER
    flat_cer=tf.keras.layers.Flatten()(bi)
    drop=tf.keras.layers.Dropout(0.3)(flat_cer)
    output_cer = tf.keras.layers.Dense(82,activation='relu',name="CER")(drop)

    #dense for COT
    # dense2 = tf.keras.layers.Dense(256,activation='relu')(conc)
    drop2=tf.keras.layers.Dropout(0.2)(flat_cer)
    output_cot = tf.keras.layers.Dense(82,activation='relu',name="COT")(drop2)

    model = tf.keras.models.Model(inputs=input_layer, outputs=[output_cot,output_cer])
    model.compile(optimizer='adam',loss=['mse','mse'],metrics=[ltype,ltype2,ltype3])
    model.summary()


    #earlystopping to find the optimal number of epochs 
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                                mode ="min", patience = 25,  
                                                restore_best_weights = True,
                                               verbose=0) 


    history = model.fit(train_image,[train_label_cot,train_label_cer],batch_size = bsize,epochs = eps,
                              callbacks =[earlystopping],validation_split=0.125)
    # test and predict ---------------------------------------------------------------------------------------
    print('test result')
    results = model.evaluate(test_image, [test_label_cot,test_label_cer])
    # results
    predictions=model.predict(test_image)

    cot_mse = results[4]
    cer_mse = results[7]
    
    model.save('saved_model/500m-multiview-BiE/model(%.f).h5'%(n+1))
    del model


    return results,cot_mse,cer_mse,history,predictions
