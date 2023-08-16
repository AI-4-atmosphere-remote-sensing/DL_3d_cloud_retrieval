


'''
    Author: Xiangyang Meng
    Email: xmeng1@umbc.edu
This code is BiLSTM model. 
'''




import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.list_physical_devices(device_type=None)
# import tensorflow.keras as keras
# from tensorflow.keras import models
# from tensorflow.keras import layers
# import matplotlib.pyplot as plt
# import numpy as np
# import h5py
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dropout,MultiHeadAttention,Bidirectional,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate
from tensorflow.keras import callbacks 
from sklearn.model_selection import KFold

# Set CUDA device explicitly
gpu = tf.config.experimental.list_physical_devices('GPU')[1] 
# print(gpu)
tf.config.experimental.set_visible_devices(gpu, 'GPU')
logical_gpus = tf.config.experimental.list_logical_devices('GPU') 
# print(logical_gpus[0])
# print(tf.config.get_visible_devices('GPU'))

# Verify correct GPU is selected
assert len(logical_gpus) == 1


os=82
ts=os
ltype = 'mean_squared_error' #evaluation metric MSE 
type2=tf.keras.metrics.RootMeanSquaredError()  #evaluation metric RMSE
bsize = 16 #batch size
eps = 4000 #maximum number of epochs
path_model="saved_model/bilstm3"





#--------
# BiLSTM3
#-------
def eval_model(train_image, test_image,train_label_cot,train_label_cer, test_label_cot, test_label_cer,n):

    ltype=tf.keras.metrics.MeanAbsolutePercentageError()
    ltype2 = 'mean_squared_error'
    ltype3=tf.keras.metrics.RootMeanSquaredError()

    bsize = 256
    eps = 4000
    
    input_layer = tf.keras.layers.Input(shape=(82,1),name='Input')
    bi=Bidirectional(tf.keras.layers.LSTM(units=64, activation='tanh',return_sequences = True,input_shape=(82,1)))(input_layer) 
    bi=Bidirectional(tf.keras.layers.LSTM(units=64, activation='tanh',return_sequences = True))(bi) 
    flat=tf.keras.layers.Flatten()(bi)
    output_cot = tf.keras.layers.Dense(82,activation='relu',name='COT')(flat) 

    #dense for CER
    flat_cer=tf.keras.layers.Flatten()(bi)
    drop=tf.keras.layers.Dropout(0.3)(flat_cer)
    output_cer = tf.keras.layers.Dense(82,activation='relu',name='CER')(drop)

   
    model = tf.keras.models.Model(inputs=input_layer, outputs=[output_cot,output_cer])
    model.compile(optimizer='adam',loss=['mse','mse'],metrics=[ltype,ltype2,ltype3])
    model.summary()


    #earlystopping to find the optimal number of epochs 
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                                mode ="min", patience = 25,  
                                                restore_best_weights = True,
                                               verbose=1) 


    history = model.fit(train_image,[train_label_cot,train_label_cer],batch_size = bsize,epochs = eps,
                              callbacks =[earlystopping],validation_split=0.125)

    # test and predict ---------------------------------------------------------------------------------------

    print('test result')
    results = model.evaluate(test_image, [test_label_cot,test_label_cer])
    results
    predictions=model.predict(test_image)

    cot_mse = results[4]
    cer_mse = results[7]
    
    # model.save('saved_model/500m-multiview-bilstm/model(%.f).h5'%(n+1))
    model.save('saved_model/500m-multiview-bilstm4/model(%.f).h5'%(n+1))
    del model


    return results,cot_mse,cer_mse,history,predictions





