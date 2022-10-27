


'''
    Author: Xiangyang Meng
    Email: xmeng1@umbc.edu

This code is for training the deep learning model of single-view COT retrieval at SZA=60 and VZA=0. The inputs are radiance and ouputs are cloud optical thickness(cot).
The datasets used for this model are:
Radiance (Input): 'data_reflectance.h5'
Cloud Optical Thickness (Ground-truth):'data_cot.h5'

They are the results from running the dataset preparing code, 'Dataset_preparation.py'. 

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
from tensorflow.keras.layers import Dropout,MultiHeadAttention,Bidirectional
from tensorflow.keras import callbacks 
from sklearn.model_selection import KFold
from utility import cross_val
import argparse





parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,help='The model you want to train')
parser.add_argument('--radiance_file_name', type=str, required=True,help='Input h5 file')
parser.add_argument('--cot_file_name', type=str, required=True,help='Output h5 file')
args = parser.parse_args()

#load model
dl_model=__import__(args.model, 'r')


#load data
hf_r = h5py.File(args.radiance_file_name, 'r')
r=hf_r['dataset_reflectance']
hf_c = h5py.File(args.cot_file_name, 'r')
c=hf_c['dataset_cot']
c=c[:]
r=r[:]
r,c= shuffle(r, c, random_state=0)

n_folds=5 #5-fold cross validation
os=82
ts=os




#train test split for five-fold cross validation-----
X_train,X_test,y_train,y_test=cross_val(r,c,ts,os,n_folds=5)
np.save('X_test_1.npy',X_test[0])
np.save('X_test_2.npy',X_test[1])
np.save('X_test_3.npy',X_test[2])
np.save('X_test_4.npy',X_test[3])
np.save('X_test_5.npy',X_test[4])
np.save('y_test_1.npy',y_test[0])
np.save('y_test_2.npy',y_test[1])
np.save('y_test_3.npy',y_test[2])
np.save('y_test_4.npy',y_test[3])
np.save('y_test_5.npy',y_test[4])




# train model-----
def main():
    for n in range(n_folds):
        train_image,train_label= X_train[n],y_train[n]
        model,history= dl_model.eval_model(train_image,train_label,n)
        
if __name__ == "__main__":
    main()











