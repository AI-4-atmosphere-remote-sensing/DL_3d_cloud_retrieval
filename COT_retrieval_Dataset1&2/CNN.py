'''
    Author: Xiangyang Meng
    Email: xmeng1@umbc.edu
This code is for single-view COT retrieval at SZA=60 and VZA=0. The inputs are radiance and ouputs are cloud optical thickness(cot).
The datasets used for this model are:
Radiance (Input): 'data_reflectance.h5'
Cloud Optical Thickness (Ground-truth):'data_cot.h5'
They are the results from running the dataset preparing code, 'Dataset preparation_d1d2.py'. 
'''


import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.list_physical_devices(device_type=None)
import tensorflow.keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import h5py
from keras import callbacks 
import keras
from sklearn.utils import shuffle
from sklearn.model_selection import KFold




num = 4000 # number of profiles
nvza = 12 
nsza=6 
os = 1 # output slice size
halo=2
ts = 1+halo*2 # input slice size 
num=4000




fname_r = "data_reflectance.h5"
hf_r = h5py.File(fname_r, 'r')
r=hf_r['dataset_reflectance']
r.shape

fname_c = "data_cot.h5"
hf_c = h5py.File(fname_c, 'r')
c=hf_c['dataset_cot']
c.shape

c=c[:]
r=r[:]
r,c= shuffle(r, c, random_state=1)




l2r=82
l2r_padding=l2r+2*2

image=np.zeros((l2r*num,ts)) 
label=np.zeros((l2r*num,os)) 

r1=np.zeros((num,l2r_padding))
c1=np.zeros((num,l2r_padding))
r1[:,2:(l2r_padding-2)]=r[:,:]
c1[:,2:(l2r_padding-2)]=c[:,:]
for i in range(num):
    r1[i,:2]=r[i,0]
    r1[i,(l2r_padding-2):]=r[i,81]


for i in range(num):
    for n in range(l2r):
        img= r1[i,n*os:n*os+ts] 
        lb = c1[i,n*os+halo:n*os+halo+os] 
        image[i*l2r+n]=img
        label[i*l2r+n]=lb 




# CNN -------------------------------------------

ltype = 'mean_squared_error'
type2=tf.keras.metrics.RootMeanSquaredError()
bsize = 32
eps = 4000

def eval_model(train_image, test_image, train_label, test_label):


    input_layer=tf.keras.layers.Input(shape=(ts,1),name='Input')
    conv1=tf.keras.layers.Conv1D(300, kernel_size=3, activation='relu', input_shape=(ts,))(input_layer) 
    conv2=tf.keras.layers.Conv1D(240, kernel_size=3, activation='relu')(conv1) 
    conv3=tf.keras.layers.Conv1D(80, kernel_size=1)(conv2)
    flat=tf.keras.layers.Flatten()(conv3)
    out=tf.keras.layers.Dense(os, activation="linear")(flat)
    model = tf.keras.models.Model(inputs=input_layer, outputs=out)
    model.compile(optimizer='adam',loss='mse',metrics=[ltype,type2])
    model.summary()
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                                mode ="min", patience = 25,  
                                                restore_best_weights = True,
                                               verbose=1) 
    history = model.fit(train_image,train_label,batch_size = bsize,epochs = eps,validation_split=0.125,
                              callbacks =[earlystopping])

    # test and predict ------
    results = model.evaluate(test_image, test_label)
    predictions=model.predict(test_image)
    mse = results[1]
    rmse = results[2]
    
    return model,rmse,mse,history,predictions,results



#train test split for 5-fold cross validation
n_folds=5
kf = KFold(n_splits=n_folds,random_state=None, shuffle=False)
ratio=int(r.shape[0]/n_folds)
X_train=np.zeros((5,l2r*ratio*4,ts)) #train: 800*4  test:800
y_train=np.zeros((5,l2r*ratio*4,os))
X_test=np.zeros((n_folds,l2r*ratio,ts))
y_test=np.zeros((n_folds,l2r*ratio,os))
count=0
for train_index, test_index in kf.split(image):
    X_train[count], X_test[count] = image[train_index], image[test_index] #r: radiance dataset
    y_train[count], y_test[count] = label[train_index], label[test_index] #c: cot dataset(label)
    count+=1




# evaluate model--------
cv_scores = list()
cv_scores2 = list()

for n in range(n_folds):
    train_image, test_image, train_label, test_label = X_train[n],X_test[n],y_train[n],y_test[n]
    model,rmse,mse,history,predictions,results= eval_model(train_image, test_image, train_label, test_label)
    print('MSE on test set in fold'+str(n+1)+' : '+str(mse))
    print('RMSE on test set in fold'+str(n+1)+' : '+str(rmse))
    cv_scores.append(mse)
    cv_scores2.append(rmse)




#quantitative result: average MSE on test set and its standard deviation
print('Estimated MSE %.4f with standard deviation (%.4f)' % (np.mean(cv_scores),np.std(cv_scores)))
print('Estimated RMSE %.4f with standard deviation (%.4f)' % (np.mean(cv_scores2),np.std(cv_scores2)))




predict1=predictions[-l2r*3:-l2r*2]
predict2=predictions[-l2r*2:-l2r]
predict3=predictions[-l2r:]




# plotting ------------------------------------------------------
fgnm = "predict_CNN"+"_"+str(len(history.history['val_loss'])-25)+'epochs'
fig, axs = plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(20,15))

#plot the third last profile
ax = axs[0]
ax.set_title(r"CNN retrieval vs True COT", fontsize=25)
ax.plot(range(82),c[num-3],alpha=0.5,color="blue")
ax.plot(range(82),predict1,alpha=0.5,color="green")
ax.legend(["True", "Predicted"],fontsize=15)
ax.set_ylabel(r"Case 3",fontsize=20) 

#plot the second last profile
ax = axs[1]
ax.plot(range(82),c[num-2].T,alpha=0.5,color="blue")
ax.plot(range(82),predict2,alpha=0.5,color="green")
ax.legend(["True", "Predicted"],fontsize=15)
ax.set_ylabel(r"Case 2",fontsize=20)  

# plot the last profile
ax = axs[2]
ax.plot(range(82),c[num-1],alpha=0.5,color="blue")
ax.plot(range(82),predict3,alpha=0.5,color="green")
ax.legend(["True", "Predicted"],fontsize=15)
ax.set_ylabel(r"Case 1",fontsize=20)  
ax.set_xlabel('X',fontsize=20)

plt.savefig(fgnm+".png",dpi=300,bbox_inches='tight')






