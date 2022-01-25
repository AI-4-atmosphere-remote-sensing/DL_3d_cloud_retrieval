#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import h5py


# In[3]:


import os
os.getcwd()


# In[10]:


fname1= "D:\\Code\\Jupyter\\climate_project\\New Data V3\\data/profile_00001.hdf5"
hf = h5py.File(fname1, 'r')
r=hf.get('reflectance')
#hf.close()


# In[11]:


hf.keys()


# In[12]:


r.shape


# two wavelengths in this dataset index 0 represents 0.865 while index 1 represents 2.13 microns.

# In[13]:


import os

path, dirs, files = next(os.walk("D:\\Code\\Jupyter\\climate_project\\New Data V3\\data"))
file_count = len(files)
file_count


# VZA’s in degrees = [ 60, 53.1, 45.5, 36.8, 25.8, 0 ,-0,- 25.8 ,- 36.8 , -45.5,- 53.1, -60]     
# 
# 	Positive VZA represents forward, and negative represents backward
# 
# 
# 
#                     index 0 corresponds to mu_VZA=0.5(i.e., VZA=60 degrees)
#                     index 1 corresponds to mu_VZA=0.6 (i.e., VZA=53.1 degrees)
#                     index 2 corresponds to mu_VZA=0.7 (i.e., VZA=45.5 degrees)
#                     index 3 corresponds to mu_VZA=0.8 (i.e., VZA=36.8 degrees)
#                     index 4 corresponds to mu_VZA=0.9 (i.e., VZA=25.8 degrees)
#                     index 5 corresponds to mu_VZA=1 (i.e., VZA=0 degrees)
#                     index 6 corresponds to mu_VZA=1 (i.e., VZA=-0 degrees)
#                     index 7 corresponds to mu_VZA=0.9 (i.e., VZA=-25.8 degrees)
#                     index 8 corresponds to mu_VZA=0.8 (i.e., VZA=-36.8 degrees)
#                     index 9 corresponds to mu_VZA=0.7 (i.e., VZA=-45.5 degrees)
#                     index 10 corresponds to mu_VZA=0.6 (i.e., VZA=-53.1 degrees)
#                     index 11 corresponds to mu_VZA=0.5 (i.e., VZA=-60 degrees)

# SZA’s in degrees =[0, 25.8, 36.9, 45.6, 53.1, 60.0]
# 
# 
# 
# 
#               index 0 corresponds to the first SZA=0
#               index 1 corresponds to the first SZA=25.8
#               index 2 corresponds to the first SZA=36.8
#               index 3 corresponds to the first SZA=45.6
#               index 4 corresponds to the first SZA=53.1
#               index 5 corresponds to the first SZA=60.0

# # average reflectance to 500m 

# In[14]:


r_data = np.empty((4000,2,12,6,4096), dtype=float) # entire reflectance dataset
cot_data=np.empty((4000,4096),dtype=float) #entire cot dataset
re_data=np.empty((4000,4096),dtype=float) #entire effective radius 
for i in range(4000):#4000 profiles
    fname = "D:\\Code\\Jupyter\\climate_project\\New Data V3\\data/profile_%05d.hdf5"%(i+1)
    hf = h5py.File(fname, 'r')
    r_data[i,:,:,:,:] = np.array(hf.get("reflectance"))
    cot_data[i,:]=np.array(hf.get('tau'))
    re_data[i,:]=np.array(hf.get('re'))
    hf.close()


# In[15]:


#use standard agnles
r=r_data


# In[16]:


r_500m=np.empty((4000,2,12,6,82), dtype=float)

for i in range(4000):
    for b in range(2):
        for a in range(12):
            for c in range(6):
                for j in range(82):#4096/500=81.92, round to 82
                    if (j+1)*50<4096:
                        r_500m[i][b][a][c][j]=np.mean(r[i][b][a][c][j*50:(j+1)*50])
                    else:
                        r_500m[i][b][a][c][j]=np.mean(r[i][b][a][c][j*50:])


# In[17]:


r_500m.shape


# In[18]:


with open('radiances82.npy', 'wb') as f:
    np.save(f,r_500m)


# In[19]:


radiance=r_500m[:,:,5,5,:] #SZA=60 VZA=0
radiance.shape


# In[35]:


with open('radiancesza60vza0_2wave_82.npy', 'wb') as f:
    np.save(f,radiance)


# In[98]:


cot_500m=np.empty((4000,82), dtype=float)


# In[99]:


for i in range(4000):
    for j in range(82):#4096/500=81.92, round to 82
        if (j+1)*50<4096:
            cot_500m[i][j]=np.mean(cot_data[i][j*50:(j+1)*50])
        else:
            cot_500m[i][j]=np.mean(cot_data[i][j*50:])


# In[103]:


with open('cotsza60vza0_2wave_82.npy', 'wb') as f:
    np.save(f,cot_500m)


# In[43]:


re_500m=np.empty((4000,82), dtype=float)


# In[44]:


for i in range(4000):
    for j in range(82):#4096/500=81.92, round to 82
        if (j+1)*50<4096:
            re_500m[i][j]=np.mean(re_data[i][j*50:(j+1)*50])
        else:
            re_500m[i][j]=np.mean(re_data[i][j*50:])


# In[21]:


with open('resza60vza0_2wave_82.npy', 'wb') as f:
    np.save(f,re_500m)


# In[ ]:





# In[14]:


import matplotlib.pyplot as plt
# plotting the profile_00001
fig,ax=plt.subplots(dpi=300)
ax.plot(range(82),radiance[1][0])
ax.plot(range(82),radiance[1][1])

ax.set_ylabel('Reflectance')
ax.set_xlabel('X')

ax.legend(['Wavelength of 0.865 microns','Wavelength of 2.13 microns'])

plt.title('Profile_00002 with 500m resolution, SZA=60 VZA=0')


# In[20]:


fig,ax=plt.subplots(dpi=300)
ax.plot(range(82),radiance[565][0])
ax.plot(range(82),radiance[565][1])

ax.set_ylabel('Reflectance')
ax.set_xlabel('X')

ax.legend(['Wavelength of 0.865 microns','Wavelength of 2.13 microns'])

plt.title('Profile_01001 with 500m resolution, SZA=60 VZA=0')


# In[17]:


fig,ax=plt.subplots(dpi=300)
ax.plot(range(82),radiance[2000][0])
ax.plot(range(82),radiance[2000][1])

ax.set_ylabel('Reflectance')
ax.set_xlabel('X')

ax.legend(['Wavelength of 0.865 microns','Wavelength of 2.13 microns'])

plt.title('Profile_02001 with 500m resolution, SZA=60 VZA=0')


# In[ ]:





# In[ ]:





# In[ ]:




