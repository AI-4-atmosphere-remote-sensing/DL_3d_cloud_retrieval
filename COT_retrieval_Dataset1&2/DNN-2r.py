'''
    Author: Xiangyang Meng
    Email: xmeng1@umbc.edu

This code is for single-view COT retrieval at SZA=60 and VZA=0. The datasets used for this model are:
Radiance (Input): 'data_reflectance.h5'
Cloud Optical Thickness (COT):"data_cot.h5"

They are the outputs from running the dataset preparing code, 'Dataset preparation.py'.

'''

import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.list_physical_devices(device_type=None)
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import h5py
from keras import callbacks 
from tensorflow.keras.layers import Input, concatenate,Dense
from keras.models import Sequential,Model
from keras.layers import Concatenate, Dense,Input, concatenate,Dropout,Flatten, Add
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from utility import cross_val

num = 4000 
nvza = 12 
nsza=6
os=82
ts=82

fname_r = "data_reflectance.h5"
hf_r = h5py.File(fname_r, 'r')
r=hf_r['dataset_reflectance']

fname_c = "data_cot.h5"
hf_c = h5py.File(fname_c, 'r')
c=hf_c['dataset_cot']

c=c[:]
r=r[:]
r,c= shuffle(r, c, random_state=0)


#------
# DNN
#------

def evaluate_model(train_image, test_image, train_label, test_label):
    input_img = Input(shape=(ts))
    hn = Dense(32, activation='relu')(input_img)
    hn1 =Dense(1024, activation='relu')(hn)
    hn1=Dense(1024, activation='relu')(hn1)
    hn1=Dropout(0.2)(hn1)
    hn1=Dense(32, activation='relu')(hn1) #
    out_both=Add()([hn, hn1])
    hn2=Dense(os, activation='linear')(out_both)
    model = Model(input_img, outputs=[hn2])
    model.summary()

    ltype = 'mean_squared_error'
    ltype2=tf.keras.metrics.RootMeanSquaredError()
    bsize = 128
    eps = 4000

    model.compile(optimizer='adam',loss=ltype,metrics=[ltype,ltype2])
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss",mode ="min", patience =25,restore_best_weights = True
                                            ,verbose=1) 
    history = model.fit(train_image,train_label,batch_size = bsize,epochs = eps,validation_split=0.125,callbacks =[earlystopping]) #validation set(10%)/trainset(80%)=0.125

    # test and predict ------
    print('test result')
    results = model.evaluate(test_image, test_label)
    predictions=model.predict(test_image)
    mse=results[1]
    rmse=results[2]
    
    return model,rmse,mse,history,predictions


#train test split for five-fold cross validation-----
X_train,X_test,y_train,y_test=cross_val(r,c,ts,os,n_folds=5)


# evaluate model-----
cv_scores = list()
cv_scores2 = list()
n_folds=5
for n in range(n_folds):
    train_image, test_image, train_label, test_label = X_train[n],X_test[n],y_train[n],y_test[n]
    model,rmse,mse,history,predictions= evaluate_model(train_image, test_image, train_label, test_label)
    print('MSE on test set in fold'+str(n+1)+' : '+str(mse))
    print('RMSE on test set in fold'+str(n+1)+' : '+str(rmse))
    cv_scores.append(mse)
    cv_scores2.append(rmse)


# quantitative result: average MSE/RMSE on test set and its standard deviation
print('Estimated MSE on testset is %.4f with standard deviation (%.4f)' % (np.mean(cv_scores),np.std(cv_scores)))
print('MSE in five folds cross validation:',cv_scores)
print('Estimated RMSE on testset is %.4f with standard deviation (%.4f)' % (np.mean(cv_scores2),np.std(cv_scores2)))
print('RMSE in five folds cross validation:',cv_scores2)


np.save('predictions_dnn.npy',predictions)




fname_c = "data_cot.h5"
hf_c = h5py.File(fname_c, 'r')
c=hf_c['dataset_cot']
c=c[:]




predicted_cot3_82=np.zeros((82), dtype=float)
path="D:\\code\\Fiona\\climate project\\New Data\\retrieved_COT"
for i in range(4000):
    if np.array_equal(test_label[-3],c[i]):
        print('profile',i+1)
        p1=c[i]
        Profile3=i+1
        fname = path+"//profile_%05d.hdf5"%(i+1)
        hf = h5py.File(fname, 'r')
        predicted_cot3=hf.get('Retrieved_tau')
        predicted_cot3=np.array(predicted_cot3)
        for j in range(82):
            if (j+1)*50<4096:
                predicted_cot3_82[j]=np.mean(predicted_cot3[j*50:(j+1)*50])
            else:
                predicted_cot3_82[j]=np.mean(predicted_cot3[j*50:])

predicted_cot2_82=np.zeros((82), dtype=float)
for i in range(4000):
    if np.array_equal(test_label[-2],c[i]):
        print('profile',i+1)
        p2=c[i]
        Profile2=i+1
        fname = path+"//profile_%05d.hdf5"%(i+1)
        hf = h5py.File(fname, 'r')
        predicted_cot2=hf.get('Retrieved_tau')
        predicted_cot2=np.array(predicted_cot2)
        for j in range(82):
            if (j+1)*50<4096:
                predicted_cot2_82[j]=np.mean(predicted_cot2[j*50:(j+1)*50])
            else:
                predicted_cot2_82[j]=np.mean(predicted_cot2[j*50:])
        
predicted_cot1_82=np.zeros((82), dtype=float)
for i in range(4000):
    if np.array_equal(test_label[-1],c[i]):
        print('profile',i+1)
        p3=c[i]
        Profile1=i+1
        fname = path+"//profile_%05d.hdf5"%(i+1)
        hf = h5py.File(fname, 'r')
        predicted_cot1=hf.get('Retrieved_tau')
        predicted_cot1=np.array(predicted_cot1)
        for j in range(82):
            if (j+1)*50<4096:
                predicted_cot1_82[j]=np.mean(predicted_cot1[j*50:(j+1)*50])
            else:
                predicted_cot1_82[j]=np.mean(predicted_cot1[j*50:])

# visualization -----
predict1=np.zeros((82)) 
predict1=predictions[-3]
predict2=np.zeros((82))
predict2=predictions[-2]
predict3=np.zeros((82))
predict3=predictions[-1]

fgnm = "plots/DNN2r_SZA=60_VZA=0"+"_"+str(len(history.history['val_loss'])-25)+'_epochs'
fig, axs = plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(20,15))

#plot the third last profile
ax = axs[0]
ax.set_title(r"DNN retrieval vs True COT", fontsize=25)
ax.plot(range(82),test_label[-3],alpha=0.8,color="blue") 
ax.plot(range(82),predict1,alpha=0.8,color="green") 
ax.legend(["True", "Predicted"],fontsize=15)
ax.set_ylabel(r"Profile_%d"%(Profile3),fontsize=20)

#plot the second last profile
ax = axs[1]
ax.plot(range(82),test_label[-2],alpha=0.8,color="blue")
ax.plot(range(82),predict2,alpha=0.8,color="green")
ax.legend(["True", "Predicted"],fontsize=15)
ax.set_ylabel(r"Profile_%d"%(Profile2),fontsize=20)  

# plot the last profile
ax = axs[2]
ax.plot(range(82),test_label[-1],alpha=0.8,color="blue")
ax.plot(range(82),predict3,alpha=0.8,color="green")
ax.legend(["True", "Predicted"],fontsize=15)
ax.set_ylabel(r"Profile_%d"%(Profile1),fontsize=20)  
ax.set_xlabel('X',fontsize=25)
