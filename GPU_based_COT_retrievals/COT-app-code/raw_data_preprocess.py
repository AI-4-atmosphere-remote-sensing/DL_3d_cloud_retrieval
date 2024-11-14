'''
   Author: Zahid Hassan Tushar
   Email: ztushar1@umbc.edu


    It is developed for preprocessing data for COT retreivals by deep learning models 
    The current dataset has 102 LES profiles at single-view SZA=60 and VZA=0.

    The data have multiple resolutions. This snippet of code only extracts data at 200m resolution.
    Part 1 is used to convert hdf files to numpy and save them.
    It also encodes the profile number and cloud mask in the data.

    Part 2 extracts patches from the profiles in numpy formats. It also creates the 
    Cross-Validated (CV) dataset. The 100 profiles are divided into five folds. Each fold has
    20 profiles. We create five datasets where each dataset has train, validation and test set.
    Three folds and profiles 101 and 102 are used as train set. The rest two folds are validation 
    and test set respectfully.

'''

# Import libraries
import numpy as np
import h5py
import os
import argparse
from utilities.cross_val import cross_val

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data_dir', type=str, default="/home/local/AD/ztushar1/Data/cloud_fraction_100LES_sza60/", help='dataset directory')
    args = parser.parse_args()
    return args


# Parse the arguments
args = parse_args()

# Read data using dataset directory path
num=102
data_dir=args.data_dir

# Create a directory for the preprocessed data
image_data_dir = "preprocessed_data"
try:
    os.makedirs(image_data_dir)
except OSError as error:
    print("Directory Exists")

'''
###############################################################################################
########################          Part 1     ##################################################
###############################################################################################
'''

# Create empty arrays for the reflectance and cot data
# reflectance data has three channels (0.66 um , 2.13 um and IPA retrievals)
r_data_full = np.empty((num,72,72,3), dtype=float) 
cot_data_full =np.empty((num,72,72,3),dtype=float) 

# Commented out due to none values
for i in range(num):
    print("Processed porfiles: ",i+1)
    # Load the data
    fname = data_dir+"/LES_profile_%05d.hdf5"%(i+1)
    hf = h5py.File(fname, 'r')
    # print(hf.keys()) 


    fname = data_dir+"COT_retrievals/"+"LES_profile_%05d.hdf5"%(i+1)
    hf2 = h5py.File(fname, 'r')
    # print(hf2.keys()) 

    r_data_full[i,:,:,0] = np.nan_to_num(np.array(hf.get("Reflectance_0p66_(200m resolution)")))
    r_data_full[i,:,:,1] = np.nan_to_num(np.array(hf.get("Reflectance_2p13_(200m resolution)")))
    r_data_full[i,:,:,2] = np.nan_to_num(np.array(hf2.get("Retrieved_COT_200m_resolution")))
    cot_data_full[i,:,:,0] = np.nan_to_num(np.array(hf.get("Cloud_optical_thickness_(200m resolution)")))

    # Cloud Mask
    cot_data_full[i,:,:,1] = np.nan_to_num(np.array(hf.get("Cloud_fraction_(200m)")))
    # Profile number
    cot_data_full[i,:,:,2] = (i+1)


    hf.close()
    hf2.close()

# Save the data
fname_r = image_data_dir + "/reflectance_200m.npy"
fname_c = image_data_dir + "/cot_200m.npy"
np.save(fname_r,r_data_full)
np.save(fname_c,cot_data_full)

'''
###############################################################################################
########################          Part 2     ##################################################
###############################################################################################
'''
processed_data_dir = image_data_dir+"/CV_dataset"
try:
    os.makedirs(processed_data_dir)
except OSError as error:
    print("Directory Exists")

X_train,X_valid,X_test,Y_train,Y_valid,Y_test=cross_val(r_data_full[0:100,:,:,:],cot_data_full[0:100,:,:,:],n_folds=5)


for i in range (5):
    fname_X_train = processed_data_dir+"/X_train_%01d.npy"%(i)
    fname_X_valid = processed_data_dir+"/X_valid_%01d.npy"%(i)
    fname_X_test = processed_data_dir+"/X_test_%01d.npy"%(i)
    fname_Y_train = processed_data_dir+"/Y_train_%01d.npy"%(i)
    fname_Y_valid = processed_data_dir+"/Y_valid_%01d.npy"%(i)
    fname_Y_test = processed_data_dir+"/Y_test_%01d.npy"%(i)

    np.save(fname_X_train,np.concatenate((X_train[i],r_data_full[100:102,:,:,:]),axis=0))
    np.save(fname_X_valid,X_valid[i])
    np.save(fname_X_test,X_test[i])
    np.save(fname_Y_train,np.concatenate((Y_train[i],cot_data_full[100:102,:,:,:]),axis=0))
    np.save(fname_Y_valid,Y_valid[i])
    np.save(fname_Y_test,Y_test[i])



print("Raw Data Processing Done!")