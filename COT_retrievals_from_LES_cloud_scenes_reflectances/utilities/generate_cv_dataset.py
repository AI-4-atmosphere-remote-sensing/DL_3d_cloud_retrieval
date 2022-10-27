'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu

    Write Comment
'''

# Import libraries
from email import header
import numpy as np

from .utilities import *
from .patch_generator import extract_patch

def generate_cv_dataset(source_dir):


    processed_data_dir = source_dir

    patch_dir=source_dir

    # Csv file is to store the dataset mean and std values which will be later used for normalization

    for i in range (5):
        # Read the numpy data
        fname_X_train = processed_data_dir+"/X_train_%01d.npy"%(i)
        fname_X_valid = processed_data_dir+"/X_valid_%01d.npy"%(i)
        fname_X_test  = processed_data_dir+"/X_test_%01d.npy"%(i)
        fname_Y_train = processed_data_dir+"/Y_train_%01d.npy"%(i)
        fname_Y_valid = processed_data_dir+"/Y_valid_%01d.npy"%(i)
        fname_Y_test  = processed_data_dir+"/Y_test_%01d.npy"%(i)

        X_train = np.load(fname_X_train)
        X_valid = np.load(fname_X_valid)
        X_test  = np.load(fname_X_test)
        Y_train = np.load(fname_Y_train)
        Y_valid = np.load(fname_Y_valid)
        Y_test  = np.load(fname_Y_test)

        # extract patches from the profiles
        X_train_patches = extract_patch(X_train,(10,10),stride=2)
        X_valid_patches = extract_patch(X_valid,(10,10),stride=2)
        X_test_patches  = extract_patch(X_test,(10,10),stride=2)
        Y_train_patches = extract_patch(Y_train,(10,10),stride=2)
        Y_valid_patches = extract_patch(Y_valid,(10,10),stride=2)
        Y_test_patches  = extract_patch(Y_test,(10,10),stride=2)

        # print("x train size: ", np.shape(X_train_patches)[0])
        # print("x valid size: ", np.shape(X_valid_patches)[0])
        # print("x test size: ", np.shape(X_test_patches)[0])


        # Save the patches
        fname_X_train = patch_dir+"/X_train_patch_%01d.npy"%(i)
        fname_X_valid = patch_dir+"/X_valid_patch_%01d.npy"%(i)
        fname_X_test  = patch_dir+"/X_test_patch_%01d.npy"%(i)
        fname_Y_train = patch_dir+"/Y_train_patch_%01d.npy"%(i)
        fname_Y_valid = patch_dir+"/Y_valid_patch_%01d.npy"%(i)
        fname_Y_test  = patch_dir+"/Y_test_patch_%01d.npy"%(i)

        np.save(fname_X_train,X_train_patches)
        np.save(fname_X_valid,X_valid_patches)
        np.save(fname_X_test,X_test_patches)
        np.save(fname_Y_train,Y_train_patches)
        np.save(fname_Y_valid,Y_valid_patches)
        np.save(fname_Y_test,Y_test_patches)


    
    print("Patches for CV dataset Generated!")

if __name__=="__main__":

    generate_cv_dataset("/home/local/AD/ztushar1/cot_retrieval/preprocessed_data/CV_dataset")

    print("Done!")