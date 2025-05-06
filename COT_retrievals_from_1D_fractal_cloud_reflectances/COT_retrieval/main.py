'''
    Notices: “Copyright © 2022 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.   All Rights Reserved.”

    Disclaimer: 
    No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
    Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.


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











