

'''
    Author: Xiangyang Meng
    Email: xmeng1@umbc.edu

This code is for training the deep learning model of single-view COT retrieval at SZA=60 and VZA=0. The inputs are radiance and ouputs are cloud optical thickness(cot).
The datasets used for this model are:
Radiance (Input): 'data_reflectance.h5'
Cloud Optical Thickness (Ground-truth):'data_cot.h5'
They are the outputs from running the dataset preparing code, 'Dataset_preparation.py'. 

To train any RNN-based model, you may import the corresponding model. For example, if you want to train BiLSTM_Transformer_embedding, please type 0.

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
from utility import cross_val


model_names=['BiLSTM_Transformer_embedding','LSTM','BiLSTM_with_Transformer','BiLSTM','BiLSTM_embedding','LSTM_embedding','LSTM_with_Transformer','Transformer']
inputnames=input("Please choose which model you want to use. Type 0 for BiLSTM_Transformer_embedding, 1 for LSTM, 2 for BiLSTM_with_Transformer, 3 for BiLSTM, 4 for BiLSTM_embedding, 5 for LSTM_embedding, 6 for LSTM_with_Transformer, 7 for Transformer")
dl_model=__import__(model_names[int(inputnames)])


fname_r = "data_reflectance.h5"
fname_c = "data_cot.h5"
num = 4000 # number of profiles
n_folds=5 #5-fold cross validation
os=82
ts=os
#load data
hf_r = h5py.File(fname_r, 'r')
r=hf_r['dataset_reflectance']
hf_c = h5py.File(fname_c, 'r')
c=hf_c['dataset_cot']
c=c[:]
r=r[:]
r,c= shuffle(r, c, random_state=0)




#train test split for five-fold cross validation-----
X_train,X_test,y_train,y_test=cross_val(r,c,ts,os,n_folds=5)
np.save('X_test.npy',X_test)
np.save('y_test.npy',y_test)



# train model-----
def main():
    for n in range(n_folds):
        train_image,train_label= X_train[n],y_train[n]
        model,history= dl_model.eval_model(train_image,train_label,n)




if __name__ == "__main__":
    main()






