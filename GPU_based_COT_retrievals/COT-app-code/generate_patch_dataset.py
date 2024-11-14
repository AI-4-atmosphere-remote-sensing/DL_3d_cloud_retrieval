'''
   Author: Zahid Hassan Tushar
   Email: ztushar1@umbc.edu

    Takes data_preparation output npy profiles of dim [102,72,72,2] as input and 
    generates patches with stride=2.


'''
# Import libraries
import numpy as np
import os
from utilities import *
# from utilities.generate_cv_dataset import generate_cv_dataset

# generate_cv_dataset
generate_cv_dataset("preprocessed_data/CV_dataset")

# generate_cloud_dataset
'''
# cloud 25
# train set total samples = 47700
# valid and test set total samples = 19050


# cloud 50
# train set total samples = 11600
# valid and test set total samples = 3700


# cloud 75
# train set total samples = 9100
# valid and test set total samples = 3150    
'''
cloud_vol_ratio = 0.25
generate_cloud_dataset("preprocessed_data/CV_dataset","preprocessed_data",cloud_vol_ratio)


    

print("done !")