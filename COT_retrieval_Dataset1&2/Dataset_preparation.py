

# '''
#     Author: Xiangyang Meng
#     Email: xmeng1@umbc.edu
# 
# This code is for preprocessing data for COT retreivals by deep learning models at single-view SZA=60 and VZA=0. The current resolution is 10m. Through this code, we can change it to new resolution of 50m or 100m.
# 
# '''



import numpy as np
import h5py




'''
current resolution: 10m =0.01km
new resolution: 500m=0.5km
Thus, we need to average every 50 reflectance values across the 4096 reflectance values in each profile.

4096/50=81.92, round to 82, in each profile.
'''




num=4000
data_size=4096
nvza=12
nsza=6
resolution=50
averaged_data=round(data_size/resolution)
path="D:\\code\\Jupyter\\climate_project\\New Data\\profiles"
r_data = np.empty((num,data_size,nsza,nvza), dtype=float) 
cot_data=np.empty((num,data_size),dtype=float) 
for i in range(0,num):
    for a in range(nsza):
        for b in range(nvza):
            fname = path+"/profile_%05d.hdf5"%(i+1)
            hf = h5py.File(fname, 'r')
            r_data[i,:,a,b] = np.array(hf.get("reflectance"))[b,a,:]
            cot=np.array(hf.get('tau'))
            cot_data[i,:]=cot
            hf.close()




#use standard agnles: sza=60 vza=0
r=r_data[:,:,5,5]




#radiance averaging
r_500m=np.empty((num,averaged_data), dtype=float)
for i in range(num):
    for j in range(averaged_data):#4096/500=81.92, round to 82
        if (j+1)*resolution<data_size:
            r_500m[i][j]=np.mean(r[i][j*resolution:(j+1)*resolution])
        else:
            r_500m[i][j]=np.mean(r[i][j*resolution:])




#cot averaging
cot_500m=np.empty((num,averaged_data), dtype=float)
for i in range(num):
    for j in range(averaged_data):#4096/500=81.92, round to 82
        if (j+1)*resolution<data_size:
            cot_500m[i][j]=np.mean(cot_data[i][j*resolution:(j+1)*resolution])
        else:
            cot_500m[i][j]=np.mean(cot_data[i][j*resolution:])




# saving to hdf5 file
hf = h5py.File('data_reflectance.h5', 'w')
hf.create_dataset('dataset_reflectance', data=r_500m)
hf.close()




hf = h5py.File('data_cot.h5', 'w')
hf.create_dataset('dataset_cot', data=cot_500m)
hf.close()




'''
Averaging reflectance to 1km
current resolution: 10m =0.01km
new resolution: 1000m=1km
So, average every 100 reflectance values across the 4096 reflectance values
And, average every 100 cot values across the 4096 cot values 

4096/100=40.96, round to 41, in each profile

'''




#radiance averaging
resolution2=100
averaged_data2=round(data_size/resolution2)
r_1000m=np.empty((num,averaged_data2), dtype=float)
for i in range(num):
    for j in range(averaged_data2):#4096/1000=40.96, round to 41
        if (j+1)*resolution2<data_size:
            r_1000m[i][j]=np.mean(r[i][j*resolution2:(j+1)*resolution2])
        else:
            r_1000m[i][j]=np.mean(r[i][j*resolution2:])




#cot averaging
cot_1000m=np.empty((num,averaged_data2), dtype=float)
for i in range(num):
    for j in range(averaged_data2):#4096/1000=40.96, round to 41
        if (j+1)*resolution2<data_size:
            cot_1000m[i][j]=np.mean(cot_data[i][j*resolution2:(j+1)*resolution2])
        else:
            cot_1000m[i][j]=np.mean(cot_data[i][j*resolution2:])




# saving to hdf5 file
hf = h5py.File('data_reflectance_1000.h5', 'w')
hf.create_dataset('data_reflectance_1000', data=r_1000m)
hf.close()




hf = h5py.File('data_cot_1000.h5', 'w')
hf.create_dataset('data_cot_1000', data=cot_1000m)
hf.close()

