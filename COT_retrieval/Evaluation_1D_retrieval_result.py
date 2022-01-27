


'''
    Author: Xiangyang Meng
    Email: xmeng1@umbc.edu

This code is for evaluating 1D retrieval results from physics method by computing MSE and RMSE of 1D retrieval results.

'''




import numpy as np
import h5py
import os


# # 1d retrieval dataset



# count how many 1d retrieval 
path="1d_rt_varying_cld"
path, dirs, files = next(os.walk(path))
file_count = len(files)
print('The number of profiles in 1D retrieval dataset:',file_count,'profiles')




#the 1d retrieval features
fname = '1d_rt_varying_cld\profile_00001.hdf5'
hf = h5py.File(fname, 'r')
ls=list(hf.keys()) 
print(ls)




path1="1d_rt_varying_cld"
retrieved_cot='Retrieved_500m_tau'
averaged_cot='Averaged_tau'
one_d_retrieval_entire=np.empty((4000,82), dtype=float) #retrieved
a_tau=np.empty((4000,82)) #truth
for i in range(4000):
    fname = path1+"\profile_%05d.hdf5"%(i+1)
    hf = h5py.File(fname, 'r')
    a=np.array(hf.get(retrieved_cot))
    a_t=np.array(hf.get(averaged_cot))
    one_d_retrieval_entire[i]=a
    a_tau[i]=a_t
    hf.close()




np.save('dataset_v2_1d_retrieval.npy',one_d_retrieval_entire)




#MSE RMSE for 1D retreival for entire COT dataset and corresponding 1D retrieval (4000 profiles)
from sklearn.metrics import mean_squared_error
import math
mse=mean_squared_error(a_tau,one_d_retrieval_entire)
rmse = math.sqrt(mse)
print('1D retrieval MSE:',mse)
print('1D retrieval RMSE:',rmse)




# new 1d retrieval in 5-fold cross validation




#MSE RMSE for 1D retreival 5-fold cross validation
from sklearn.metrics import mean_squared_error
import math
#1st fold 
mse=mean_squared_error(a_tau[:800],one_d_retrieval_entire[:800])
rmse = math.sqrt(mse)
print('1D retrieval MSE:',mse)
print('1D retrieval RMSE:',rmse)




#2nd fold 
mse1=mean_squared_error(a_tau[800:1600],one_d_retrieval_entire[800:1600])
rmse1 = math.sqrt(mse1)
print('1D retrieval MSE:',mse1)
print('1D retrieval RMSE:',rmse1)




#3rd fold 
mse2=mean_squared_error(a_tau[1600:2400],one_d_retrieval_entire[1600:2400])
rmse2 = math.sqrt(mse2)
print('1D retrieval MSE:',mse2)
print('1D retrieval RMSE:',rmse2)




#4th fold 
mse3=mean_squared_error(a_tau[2400:3200],one_d_retrieval_entire[2400:3200])
rmse3 = math.sqrt(mse3)
print('1D retrieval MSE:',mse3)
print('1D retrieval RMSE:',rmse3)




#5th fold 
mse4=mean_squared_error(a_tau[3200:],one_d_retrieval_entire[3200:])
rmse4 = math.sqrt(mse4)
print('1D retrieval MSE:',mse4)
print('1D retrieval RMSE:',rmse4)




#MSE mean
np.mean((mse,mse1,mse2,mse3,mse4))




#MSE std
np.std((mse,mse1,mse2,mse3,mse4))




#RMSE mean
np.mean((rmse,rmse1,rmse2,rmse3,rmse4))




#RMSE std
np.std((rmse,rmse1,rmse2,rmse3,rmse4))

