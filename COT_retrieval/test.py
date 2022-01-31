


'''
    Author: Xiangyang Meng
    Email: xmeng1@umbc.edu

This code is for testing the trained RNN-based model of single-view COT retrieval at SZA=60 and VZA=0, and visualizing the testing results. 

The dataset used for this model are:
Cloud Optical Thickness (Ground-truth):'data_cot.h5'
It is the output from running the dataset preparing code, 'dataset preparation.py'. 

'''




import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import h5py
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from utility import cross_val
import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--cot_file_name', type=str, required=True,help='Output (predicted COT) h5 file')
parser.add_argument('--path_1d_retrieval', type=str, required=True,help='1D retrieval files path')
parser.add_argument('--path_model', type=str, required=True,help='trained model path')
parser.add_argument('--path_predictions', type=str, required=True,help='predicted COT values saved in a numpy file')
parser.add_argument('--radiance_test', type=str, required=True,help='radiance test data')
parser.add_argument('--cot_test', type=str, required=True,help='COT test data')
parser.add_argument('--path_plots', type=str, required=True,help='sample visualized results plots path')
args = parser.parse_args()





num=4000
os=82

hf_c = h5py.File(args.cot_file_name, 'r')
c=hf_c['dataset_cot']
c=c[:]

#load test data-----
test_image=np.load(args.radiance_test)
test_label=np.load(args.cot_test)
test_sample=test_label[-1]




new_model = tf.keras.models.load_model(args.path_model)
loss,mse,rmse=new_model.evaluate(test_image, test_label)
predictions=new_model.predict(test_image)
np.save(args.path_predictions+'predictions.npy',predictions)
print('MSE on this test set:', mse)
print('RMSE on this test set:', rmse)




predicted_1d_82=np.zeros((os), dtype=float)
for i in range(num):
    if np.array_equal(test_sample,c[i]):
        print('The visualized sample test profile is profile',i+1)
        profile_number=i+1
        fname = args.path_1d_retrieval+"//profile_%05d.hdf5"%(i+1)
        hf = h5py.File(fname, 'r')
        predicted_1d=hf.get('Retrieved_tau')
        predicted_1d=np.array(predicted_1d)
        for j in range(os):
            if (j+1)*50<4096:
                predicted_1d_82[j]=np.mean(predicted_1d[j*50:(j+1)*50])
            else:
                predicted_1d_82[j]=np.mean(predicted_1d[j*50:])




predict=np.zeros((os))
predict=predictions[-1]
# visualization of comparing original COT and deep learning retrieved COT -----
figure(figsize=(20, 8), dpi=80)
plt.title('True COT vs Predicted COT',fontsize=25)
plt.plot(range(82),test_sample,alpha=0.8,color="blue") 
plt.plot(range(82),predict,alpha=0.8,color="green") 
plt.legend(["True", "Predicted"],fontsize=15)
plt.ylabel(r"Profile_%d"%(profile_number),fontsize=20)
plt.xlabel('X',fontsize=20)
plt.savefig(args.path_plots+"comparing_original_and_DL_retrieved.png",dpi=300,bbox_inches='tight')




# visualization of comparing among 1D retrieval result(physics method) deep learning retrieved COT and original COT ------
figure(figsize=(20, 8), dpi=80)
plt.title('True COT vs 3D Retrieval vs 1D Retrieval',fontsize=25)
plt.plot(range(82),test_sample,alpha=0.8,color="blue") 
plt.plot(range(82),predict,alpha=0.8,color="green") 
plt.plot(range(82),predicted_1d_82,alpha=0.8,color="red")
plt.legend(["True", "3D",'1D'],fontsize=15)
plt.ylabel(r"Profile_%d"%(profile_number),fontsize=20)
plt.xlabel('X',fontsize=20)
plt.savefig(args.path_plots+"comparing_with_1d_retrieval.png",dpi=300,bbox_inches='tight')






