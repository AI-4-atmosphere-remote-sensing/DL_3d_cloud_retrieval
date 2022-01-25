


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.list_physical_devices(device_type=None)

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




import os
os.getcwd()




r=np.load('radiancesza60vza0_2wave_82.npy')
c=np.load('cotza60vza0_2wave_82.npy')
re=np.load('resza60vza0_2wave_82.npy')




r=r.reshape(4000,82,2)




c.shape




re.shape




r,c,re= shuffle(r,c,re, random_state=0)#Shuffle arrays in a consistent way.

num=4000
#train test split
trp = 0.8; # 80 percent to train, includes validation
train_size=int(trp*num)
test_size = num - train_size #20% for testing:0.20*4000=800

print("train_size:",train_size,'profiles')
print("test_size:",test_size,'profiles')




keras.utils.plot_model(model, show_shapes=True,show_layer_names=True,expand_nested=True,rankdir='TB')





def eval_model(train_image, test_image,train_label_re,train_label_cot, test_label_re, test_label_cot):



    ltype=tf.keras.metrics.MeanAbsolutePercentageError()
    ltype2 = 'mean_squared_error'
    ltype3=tf.keras.metrics.RootMeanSquaredError()


    bsize = 16
    eps = 4000

    #bi-LSTM
    input_layer = tf.keras.layers.Input(shape=(82,2),name='Input')


    projection=tf.keras.layers.Dense(96,activation='relu')(input_layer)
    positions = tf.range(start=0, limit=82, delta=1)
    embedding=tf.keras.layers.Embedding(input_dim=(96),output_dim=(96),input_length=32)(positions)
        # adds a learnable position embedding to the projected vector
    encoded = projection+embedding

    bi=Bidirectional(tf.keras.layers.LSTM(units=64, activation='tanh',return_sequences = True))(encoded) 
    bi=Bidirectional(tf.keras.layers.LSTM(units=64, activation='tanh',return_sequences = True))(bi)



    #transformer block 
    slf_attn = MultiHeadAttention(num_heads=3,key_dim=3)(bi,bi) 
    add=tf.keras.layers.Add()([slf_attn,bi])
    layernorm1=tf.keras.layers.LayerNormalization(epsilon=1e-6)(add)
    dense1=tf.keras.layers.Dense(128,activation='relu')(layernorm1)
    add2=tf.keras.layers.Add()([layernorm1,dense1])
    layernorm2=tf.keras.layers.LayerNormalization(epsilon=1e-6)(add2)

    flat=tf.keras.layers.Flatten()(layernorm2)

    #dense for CER
    flat_cer=tf.keras.layers.Flatten()(bi)
    drop=tf.keras.layers.Dropout(0.3)(flat_cer)
    output_cer = tf.keras.layers.Dense(82,activation='relu')(drop)

    #dense for COT
    # dense2 = tf.keras.layers.Dense(256,activation='relu')(conc)
    drop2=tf.keras.layers.Dropout(0.2)(flat)
    output_cot = tf.keras.layers.Dense(82,activation='relu')(drop2)

    model = tf.keras.models.Model(inputs=input_layer, outputs=[output_cer,output_cot])
    model.compile(optimizer='adam',loss=['mse','mse'],metrics=[ltype,ltype2,ltype3])
    model.summary()


    #earlystopping to find the optimal number of epochs 
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                                mode ="min", patience = 25,  
                                                restore_best_weights = True,
                                               verbose=1) 
    # Create a callback that saves the model's weights
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                        #         save_weights_only=False,
                                           #      verbose=1,save_freq=5*bsize)


    history = model.fit(train_image,[train_label_re,train_label_cot],batch_size = bsize,epochs = eps,validation_split=0.125,
                        callbacks =[earlystopping])

    # test and predict ---------------------------------------------------------------------------------------

    print('test result')
    results = model.evaluate(test_image, [test_label_re,test_label_cot])
    results
    predictions=model.predict(test_image)

    total_loss=results[0]
    cer_mse_loss = results[1]
    cot_mse_loss = results[2]
    CER_Test_MAPE=results[3]
    CER_Test_MSE=results[4]
    CER_Test_RMSE=results[5]
    COT_Test_MAPE=results[6]
    COT_Test_MSE=results[7]
    COT_Test_RMSE=results[8]

    return model,results,total_loss,cer_mse_loss,cot_mse_loss,CER_Test_MAPE,CER_Test_MSE,CER_Test_RMSE,COT_Test_MAPE,COT_Test_MSE,COT_Test_RMSE,history,predictions


# In[54]:


n_folds=5

kf = KFold(n_splits=n_folds,random_state=None, shuffle=False)
print(kf.get_n_splits(r))
print(kf)
ratio=int(r.shape[0]/n_folds)
print(ratio)

X_train=np.zeros((5,ratio*4,82,2)) 
y_train_cer=np.zeros((5,ratio*(n_folds-1),82))
y_train_cot=np.zeros((5,ratio*(n_folds-1),82))

X_test=np.zeros((n_folds,ratio,82,2))
y_test_cer=np.zeros((n_folds,ratio,82))
y_test_cot=np.zeros((n_folds,ratio,82))

count=0
for train_index, test_index in kf.split(r):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train[count], X_test[count] = r[train_index], r[test_index] #r: radiance dataset
    y_train_cer[count],y_train_cot[count], y_test_cer[count],y_test_cot[count] = re[train_index],c[train_index],re[test_index],c[test_index] #c: cot dataset(label)
    count+=1


# In[51]:


from sklearn.preprocessing import StandardScaler


# In[55]:


# evaluate model----------------------------------------------------------------------------------------------
cer_mse_cv_scores = list()
cot_mse_cv_scores = list()
results_cv_scores = list()
total=list()
cer_mape=list()
cer_rmse=list()
cer_mse=list()
cot_mape=list()
cot_mse=list()
cot_rmse=list()


for n in range(n_folds):
        # split data    
    train_image, test_image, train_label_re,train_label_cot, test_label_re, test_label_cot = X_train[n],X_test[n],y_train_cer[n],y_train_cot[n],y_test_cer[n],y_test_cot[n]
    train_image=train_image.reshape(3200,82*2)
    test_image=test_image.reshape(800,82*2)
    scaler = StandardScaler()
    train_image = scaler.fit_transform(train_image)
    test_image = scaler.transform(test_image)
    train_image=train_image.reshape(3200,82,2)
    test_image=test_image.reshape(800,82,2)
        # evaluate model
    model,results,total_loss,cer_mse_loss,cot_mse_loss,CER_Test_MAPE,CER_Test_MSE,CER_Test_RMSE,COT_Test_MAPE,COT_Test_MSE,COT_Test_RMSE,history,predictions= eval_model(train_image, test_image,train_label_re,train_label_cot, test_label_re, test_label_cot)
    
    print('Fold',n+1,' Test Loss:',results[0])
    print('Fold',n+1,'CER Test Loss:',results[1])
    print('Fold',n+1,'COT Test Loss:',results[2])
    
    print('Fold',n+1,'CER Test MAPE:',results[3])
    print('Fold',n+1,'CER Test MSE:',results[4])
    print('Fold',n+1,'CER Test RMSE:',results[5])
    
    print('Fold',n+1,'COT Test MAPE:',results[6])
    print('Fold',n+1,'COT Test MSE:',results[7])
    print('Fold',n+1,'COT Test RMSE:',results[8])
    
    total.append(total_loss)
    cer_mape.append(CER_Test_MAPE)
    cer_mse.append(CER_Test_MSE)
    cer_rmse.append(CER_Test_RMSE)
    cot_mape.append(COT_Test_MAPE)
    cot_mse.append(COT_Test_MSE)
    cot_rmse.append(COT_Test_RMSE)
    cer_mse_cv_scores.append(cer_mse_loss)
    cot_mse_cv_scores.append(cot_mse_loss)
    results_cv_scores.append(results)


# In[56]:



total


# In[2]:


import numpy as np
np.mean([0.3529682457447052,
 0.3889194428920746,
 0.3542912006378174,
 0.3395749628543854,
 0.4096250832080841])


# In[57]:


cer_mse


# In[3]:


np.mean([0.0072343964129686356,
 0.008686362765729427,
 0.01034444198012352,
 0.006996344309300184,
 0.007759868633002043])


# In[58]:


cot_mse


# In[4]:


np.mean([0.3457338809967041,
 0.3802330493927002,
 0.34394681453704834,
 0.3325786292552948,
 0.4018651843070984])


# In[59]:


cer_mape


# In[5]:


np.mean([0.7540177702903748,
 0.8550044298171997,
 0.9260680675506592,
 0.7495445013046265,
 0.7795014381408691])


# In[60]:


cot_mape


# In[6]:


np.mean([3.288017511367798,
 3.544426679611206,
 3.2595462799072266,
 3.3338286876678467,
 3.547499179840088])


# In[61]:


with open('predictions_normalize_mape.npy', 'wb') as f:
    np.save(f,np.array(predictions))


# In[ ]:





# In[62]:


patience=25
# plotting training and validation history
plt.figure(figsize=(15,10))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('Epochs')
plt.ylabel('Loss') 


plt.title('Train and Validation Loss with '+str(len(history.history['val_loss']))+'epochs')
plt.legend()
 

plt.savefig('train_validation_after_'+str(len(history.history['val_loss'])-patience)+"_epochs.png")

print('The optimal number of epochs is '+str(len(history.history['val_loss'])-patience))


# In[63]:


predict1_cer=np.zeros((82))
predict1_cer=predictions[0][-3]
    
predict2_cer=np.zeros((82))
predict2_cer=predictions[0][-2]
    
predict3_cer=np.zeros((82))
predict3_cer=predictions[0][-1]


# In[64]:


# plotting ---------------------------------------------------------------------------------------------------
fgnm = "plots/predict_Bi-LSTM with Transformer SZA=60 VZA=0_CER"+str(len(history.history['loss'])-patience)+'_epochs'
fig, axs = plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(20,15))

# plots the last three test profiles

#plot the third last profile
ax = axs[0]
ax.set_title(r"Bi-LSTM with transformer CER retrieval vs True CER",fontsize=25)
ax.plot(range(82),test_label_re[-3],alpha=0.5,color="blue") #ground truth
ax.plot(range(82),predict1_cer,alpha=0.5,color="green") 
ax.legend(["True", "3D retrieval",'1D retrieval'],fontsize=15)
ax.set_ylabel(r"Profile_01375",fontsize=20) 

#plot the second last profile
ax = axs[1]
ax.plot(range(82),test_label_re[-2],alpha=0.5,color="blue")
ax.plot(range(82),predict2_cer,alpha=0.5,color="green")
ax.legend(["True", "3D retrieval",'1D retrieval'],fontsize=15)
ax.set_ylabel(r"Profile_01090",fontsize=20)  

# plot the last profile
ax = axs[2]
ax.plot(range(82),test_label_re[-1].T,alpha=0.5,color="blue")
ax.plot(range(82),predict3_cer,alpha=0.5,color="green")
ax.legend(["True", "3D retrieval",'1D retrieval'],fontsize=15)
ax.set_ylabel(r"Profile_03104",fontsize=20)  
ax.set_xlabel('X',fontsize=20)

plt.savefig(fgnm+".png",dpi=300,bbox_inches='tight')


# In[ ]:




