'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu

    Write Comment
'''

# Import libraries
import numpy as np
import os
import random
from .utilities import *


def generate_cloud_dataset(cv_data_dir,source_dir,cloud_vol_ratio):

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



    train_set_limit      = 47700
    valid_test_set_limit = 19050
    
    cv_patches_dir=cv_data_dir
    saved_patch_dir=source_dir+"/Cloud_"+str(np.int16(cloud_vol_ratio*100))
    cloud_vol_ratio = cloud_vol_ratio

    try:
        os.makedirs(saved_patch_dir)
    except OSError as error:
        print("Directory Exists")

    
    for i in range (5):
        fname_X_train = cv_patches_dir+"/X_train_patch_%01d.npy"%(i)
        fname_X_valid = cv_patches_dir+"/X_valid_patch_%01d.npy"%(i)
        fname_X_test  = cv_patches_dir+"/X_test_patch_%01d.npy"%(i)
        fname_Y_train = cv_patches_dir+"/Y_train_patch_%01d.npy"%(i)
        fname_Y_valid = cv_patches_dir+"/Y_valid_patch_%01d.npy"%(i)
        fname_Y_test  = cv_patches_dir+"/Y_test_patch_%01d.npy"%(i)

        X_train = np.load(fname_X_train)
        X_valid = np.load(fname_X_valid)
        X_test  = np.load(fname_X_test)
        Y_train = np.load(fname_Y_train)
        Y_valid = np.load(fname_Y_valid)
        Y_test  = np.load(fname_Y_test)

        # train set
        #  Compute cloud fraction

        cloud_fraction = np.empty((np.shape(Y_train)[0]),dtype=float)
        for k in range(np.shape(Y_train)[0]):
            cloud_fraction[k] = compute_cloud_fraction(Y_train[k,:,:,1],0.24)
        
        aa = (cloud_fraction>=cloud_vol_ratio)*1
        print("Total Samples that crossed the threshold: ",np.sum(aa))

        # Select samples based on cloud fraction
        X_train_cloud_patches = np.empty((np.sum(aa),np.shape(X_train)[1],np.shape(X_train)[2],np.shape(X_train)[3]),dtype=float)
        Y_train_cloud_patches = np.empty((np.sum(aa),np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]),dtype=float)
        idx=0
        for k in range(len(aa)):
            if aa[k]==1:
                X_train_cloud_patches[idx,:,:,:] = X_train[k,:,:,:]
                Y_train_cloud_patches[idx,:,:,:] = Y_train[k,:,:,:]
                idx+=1

        '''
        This block randomly selects samples from the cloud patches and flips them and then add them to the list to make the 
        train set have equal number of sample patches throughout the folds.
        '''
        # get how many samples needs to be randomly selected
        num = train_set_limit-np.sum(aa)

        # generate random index to select samples without replacedment 
        oversampled_idx = random.sample(range(1, np.sum(aa)), num)

        # create an empty array to store the new patches
        temp_x = np.empty((num,np.shape(X_train)[1],np.shape(X_train)[2],np.shape(X_train)[3]), dtype=float) 
        temp_y = np.empty((num,np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]), dtype=float) 
        count = 0
        for os_idx in oversampled_idx:
            temp_x[count,:,:,:] = np.fliplr(X_train_cloud_patches[os_idx,:,:,:])
            temp_y[count,:,:,:] = np.fliplr(Y_train_cloud_patches[os_idx,:,:,:])
            count+=1
        X_train_cloud_patches = np.concatenate((X_train_cloud_patches,temp_x),axis=0)
        Y_train_cloud_patches = np.concatenate((Y_train_cloud_patches,temp_y),axis=0)

        print("train set size: ",X_train_cloud_patches.shape[0])
        print("train set size: ",Y_train_cloud_patches.shape[0])
        print("train samples added: ",num)


        # Valid set
        #  Compute cloud fraction
        cloud_fraction = np.empty((np.shape(Y_valid)[0]),dtype=float)
        for k in range(np.shape(Y_valid)[0]):
            cloud_fraction[k] = compute_cloud_fraction(Y_valid[k,:,:,1],0.24)
        
        bb = (cloud_fraction>=cloud_vol_ratio)*1
        print("Total Samples that crossed the threshold: ",np.sum(bb))

        X_valid_cloud_patches = np.empty((np.sum(bb),np.shape(X_valid)[1],np.shape(X_valid)[2],np.shape(X_valid)[3]),dtype=float)
        Y_valid_cloud_patches = np.empty((np.sum(bb),np.shape(Y_valid)[1],np.shape(Y_valid)[2],np.shape(Y_valid)[3]),dtype=float)

        idx=0
        for k in range(len(bb)):
            if bb[k]==1:
                X_valid_cloud_patches[idx,:,:,:] = X_valid[k,:,:,:]
                Y_valid_cloud_patches[idx,:,:,:] = Y_valid[k,:,:,:]
                idx+=1


        '''
        This block randomly selects samples from the cloud patches and flips them and then add them to the list to make the 
        valid set have equal number of sample patches throughout the folds.
        '''
        # get how many samples needs to be randomly selected
        num = valid_test_set_limit-np.sum(bb)
        patch_ratio = num/np.sum(bb)



        # when 1x~2x samples need to be oversampled. fliplr+flipud
        if patch_ratio>1 and patch_ratio<2:
            # generate random index to select samples without replacedment 
            oversampled_idx = random.sample(range(0, np.sum(bb)), np.sum(bb))

            # create an empty array to store the new patches
            temp_x = np.empty((np.sum(bb),np.shape(X_test)[1],np.shape(X_test)[2],np.shape(X_test)[3]), dtype=float) 
            temp_y = np.empty((np.sum(bb),np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]), dtype=float) 
            count = 0
            for os_idx in oversampled_idx:
                temp_x[count,:,:,:] = np.fliplr(X_valid_cloud_patches[os_idx,:,:,:])
                temp_y[count,:,:,:] = np.fliplr(Y_valid_cloud_patches[os_idx,:,:,:])
                count+=1
            X_valid_cloud_patches = np.concatenate((X_valid_cloud_patches,temp_x),axis=0)
            Y_valid_cloud_patches = np.concatenate((Y_valid_cloud_patches,temp_y),axis=0)

            # generate random index to select samples without replacedment 
            oversampled_idx = random.sample(range(0, np.sum(bb)), num-np.sum(bb))

            # create an empty array to store the new patches
            temp_x = np.empty((num-np.sum(bb),np.shape(X_test)[1],np.shape(X_test)[2],np.shape(X_test)[3]), dtype=float) 
            temp_y = np.empty((num-np.sum(bb),np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]), dtype=float) 
            count = 0
            for os_idx in oversampled_idx:
                temp_x[count,:,:,:] = np.flipud(X_valid_cloud_patches[os_idx,:,:,:])
                temp_y[count,:,:,:] = np.flipud(Y_valid_cloud_patches[os_idx,:,:,:])
                count+=1
            X_valid_cloud_patches = np.concatenate((X_valid_cloud_patches,temp_x),axis=0)
            Y_valid_cloud_patches = np.concatenate((Y_valid_cloud_patches,temp_y),axis=0)

            # oversampled_idx = np.concatenate((oversampled_idx,random.sample(range(0, np.sum(bb)), num-np.sum(bb))),axis=0)

        # when more than 2x samples need to be oversampled. fliplr+flipud
        elif patch_ratio>2:
            # generate random index to select samples without replacedment 
            oversampled_idx = random.sample(range(0, np.sum(bb)), np.sum(bb))

            # create an empty array to store the new patches
            temp_x = np.empty((np.sum(bb),np.shape(X_test)[1],np.shape(X_test)[2],np.shape(X_test)[3]), dtype=float) 
            temp_y = np.empty((np.sum(bb),np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]), dtype=float) 
            count = 0
            for os_idx in oversampled_idx:
                temp_x[count,:,:,:] = np.fliplr(X_valid_cloud_patches[os_idx,:,:,:])
                temp_y[count,:,:,:] = np.fliplr(Y_valid_cloud_patches[os_idx,:,:,:])
                count+=1
            X_valid_cloud_patches = np.concatenate((X_valid_cloud_patches,temp_x),axis=0)
            Y_valid_cloud_patches = np.concatenate((Y_valid_cloud_patches,temp_y),axis=0)

            # generate random index to select samples without replacedment 
            oversampled_idx = random.sample(range(0, np.sum(bb)), np.sum(bb))

            # create an empty array to store the new patches
            temp_x = np.empty((np.sum(bb),np.shape(X_test)[1],np.shape(X_test)[2],np.shape(X_test)[3]), dtype=float) 
            temp_y = np.empty((np.sum(bb),np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]), dtype=float) 
            count = 0
            for os_idx in oversampled_idx:
                temp_x[count,:,:,:] = np.flipud(X_valid_cloud_patches[os_idx,:,:,:])
                temp_y[count,:,:,:] = np.flipud(Y_valid_cloud_patches[os_idx,:,:,:])
                count+=1
            X_valid_cloud_patches = np.concatenate((X_valid_cloud_patches,temp_x),axis=0)
            Y_valid_cloud_patches = np.concatenate((Y_valid_cloud_patches,temp_y),axis=0)

            oversampled_idx = random.sample(range(0, np.sum(bb)), num-2*np.sum(bb))
            

            # create an empty array to store the new patches
            temp_x = np.empty((num-2*np.sum(bb),np.shape(X_test)[1],np.shape(X_test)[2],np.shape(X_test)[3]), dtype=float) 
            temp_y = np.empty((num-2*np.sum(bb),np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]), dtype=float) 
            count = 0
            for os_idx in oversampled_idx:
                temp_x[count,:,:,:] = X_valid_cloud_patches[os_idx,:,:,:]
                temp_y[count,:,:,:] = Y_valid_cloud_patches[os_idx,:,:,:]
                count+=1
            X_valid_cloud_patches = np.concatenate((X_valid_cloud_patches,temp_x),axis=0)
            Y_valid_cloud_patches = np.concatenate((Y_valid_cloud_patches,temp_y),axis=0)


        # when less than 1x samples needs to be oversampled fliplr
        else:
            # generate random index to select samples without replacedment 
            oversampled_idx = random.sample(range(0, np.sum(bb)), num)
            # create an empty array to store the new patches
            temp_x = np.empty((num,np.shape(X_train)[1],np.shape(X_train)[2],np.shape(X_train)[3]), dtype=float) 
            temp_y = np.empty((num,np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]), dtype=float) 
            count = 0
            for os_idx in oversampled_idx:
                temp_x[count,:,:,:] = np.fliplr(X_valid_cloud_patches[os_idx,:,:,:])
                temp_y[count,:,:,:] = np.fliplr(Y_valid_cloud_patches[os_idx,:,:,:])
                count+=1
            X_valid_cloud_patches = np.concatenate((X_valid_cloud_patches,temp_x),axis=0)
            Y_valid_cloud_patches = np.concatenate((Y_valid_cloud_patches,temp_y),axis=0)

        print("valid set size: ",X_valid_cloud_patches.shape[0])
        print("valid set size: ",Y_valid_cloud_patches.shape[0])
        print("valid samples added: ",num)



        # For test set
        #  Compute cloud fraction
        cloud_fraction = np.empty((np.shape(Y_test)[0]),dtype=float)
        for k in range(np.shape(Y_test)[0]):
            cloud_fraction[k] = compute_cloud_fraction(Y_test[k,:,:,1],0.24)
        
        cc = (cloud_fraction>=cloud_vol_ratio)*1
        print("Total Samples that crossed the threshold: ",np.sum(cc))

        X_test_cloud_patches = np.empty((np.sum(cc),np.shape(X_test)[1],np.shape(X_test)[2],np.shape(X_test)[3]),dtype=float)
        Y_test_cloud_patches = np.empty((np.sum(cc),np.shape(Y_test)[1],np.shape(Y_test)[2],np.shape(Y_test)[3]),dtype=float)

        idx=0
        for k in range(len(cc)):
            if cc[k]==1:
                X_test_cloud_patches[idx,:,:,:] = X_test[k,:,:,:]
                Y_test_cloud_patches[idx,:,:,:] = Y_test[k,:,:,:]
                idx+=1

        '''
        This block randomly selects samples from the cloud patches and flips them and then add them to the list to make the 
        test set have equal number of sample patches throughout the folds.
        '''
        # get how many samples needs to be randomly selected
        num = valid_test_set_limit-np.sum(cc)

        patch_ratio = num/np.sum(cc)

        # when 1x~2x samples need to be oversampled. fliplr+flipud
        if patch_ratio>1 and patch_ratio<2:
            # generate random index to select samples without replacedment 
            oversampled_idx = random.sample(range(0, np.sum(cc)), np.sum(cc))

            # create an empty array to store the new patches
            temp_x = np.empty((np.sum(cc),np.shape(X_test)[1],np.shape(X_test)[2],np.shape(X_test)[3]), dtype=float) 
            temp_y = np.empty((np.sum(cc),np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]), dtype=float) 
            count = 0
            for os_idx in oversampled_idx:
                temp_x[count,:,:,:] = np.fliplr(X_test_cloud_patches[os_idx,:,:,:])
                temp_y[count,:,:,:] = np.fliplr(Y_test_cloud_patches[os_idx,:,:,:])
                count+=1
            X_test_cloud_patches = np.concatenate((X_test_cloud_patches,temp_x),axis=0)
            Y_test_cloud_patches = np.concatenate((Y_test_cloud_patches,temp_y),axis=0)

            # generate random index to select samples without replacedment 
            oversampled_idx = random.sample(range(0, np.sum(cc)), num-np.sum(cc))

            # create an empty array to store the new patches
            temp_x = np.empty((num-np.sum(cc),np.shape(X_test)[1],np.shape(X_test)[2],np.shape(X_test)[3]), dtype=float) 
            temp_y = np.empty((num-np.sum(cc),np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]), dtype=float) 
            count = 0
            for os_idx in oversampled_idx:
                temp_x[count,:,:,:] = np.flipud(X_test_cloud_patches[os_idx,:,:,:])
                temp_y[count,:,:,:] = np.flipud(Y_test_cloud_patches[os_idx,:,:,:])
                count+=1
            X_test_cloud_patches = np.concatenate((X_test_cloud_patches,temp_x),axis=0)
            Y_test_cloud_patches = np.concatenate((Y_test_cloud_patches,temp_y),axis=0)

            # oversampled_idx = np.concatenate((oversampled_idx,random.sample(range(0, np.sum(bb)), num-np.sum(bb))),axis=0)

        # when more than 2x samples need to be oversampled. fliplr+flipud
        elif patch_ratio>2:
            # generate random index to select samples without replacedment 
            oversampled_idx = random.sample(range(0, np.sum(cc)), np.sum(cc))

            # create an empty array to store the new patches
            temp_x = np.empty((np.sum(cc),np.shape(X_test)[1],np.shape(X_test)[2],np.shape(X_test)[3]), dtype=float) 
            temp_y = np.empty((np.sum(cc),np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]), dtype=float) 
            count = 0
            for os_idx in oversampled_idx:
                temp_x[count,:,:,:] = np.fliplr(X_test_cloud_patches[os_idx,:,:,:])
                temp_y[count,:,:,:] = np.fliplr(Y_test_cloud_patches[os_idx,:,:,:])
                count+=1
            X_test_cloud_patches = np.concatenate((X_test_cloud_patches,temp_x),axis=0)
            Y_test_cloud_patches = np.concatenate((Y_test_cloud_patches,temp_y),axis=0)

            # generate random index to select samples without replacedment 
            oversampled_idx = random.sample(range(0, np.sum(cc)), np.sum(cc))

            # create an empty array to store the new patches
            temp_x = np.empty((np.sum(cc),np.shape(X_test)[1],np.shape(X_test)[2],np.shape(X_test)[3]), dtype=float) 
            temp_y = np.empty((np.sum(cc),np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]), dtype=float) 
            count = 0
            for os_idx in oversampled_idx:
                temp_x[count,:,:,:] = np.flipud(X_test_cloud_patches[os_idx,:,:,:])
                temp_y[count,:,:,:] = np.flipud(Y_test_cloud_patches[os_idx,:,:,:])
                count+=1
            X_test_cloud_patches = np.concatenate((X_test_cloud_patches,temp_x),axis=0)
            Y_test_cloud_patches = np.concatenate((Y_test_cloud_patches,temp_y),axis=0)


            oversampled_idx = random.sample(range(0, np.sum(cc)), num-2*np.sum(cc))

            # create an empty array to store the new patches
            temp_x = np.empty((num-2*np.sum(cc),np.shape(X_test)[1],np.shape(X_test)[2],np.shape(X_test)[3]), dtype=float) 
            temp_y = np.empty((num-2*np.sum(cc),np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]), dtype=float) 
            count = 0
            for os_idx in oversampled_idx:
                temp_x[count,:,:,:] = X_test_cloud_patches[os_idx,:,:,:]
                temp_y[count,:,:,:] = Y_test_cloud_patches[os_idx,:,:,:]
                count+=1
            X_test_cloud_patches = np.concatenate((X_test_cloud_patches,temp_x),axis=0)
            Y_test_cloud_patches = np.concatenate((Y_test_cloud_patches,temp_y),axis=0)


        # when less than 1x samples needs to be oversampled fliplr
        else:
            # generate random index to select samples without replacedment 
            oversampled_idx = random.sample(range(0, np.sum(cc)), num)
            # create an empty array to store the new patches
            temp_x = np.empty((num,np.shape(X_train)[1],np.shape(X_train)[2],np.shape(X_train)[3]), dtype=float) 
            temp_y = np.empty((num,np.shape(Y_train)[1],np.shape(Y_train)[2],np.shape(Y_train)[3]), dtype=float) 
            count = 0
            for os_idx in oversampled_idx:
                temp_x[count,:,:,:] = np.fliplr(X_test_cloud_patches[os_idx,:,:,:])
                temp_y[count,:,:,:] = np.fliplr(Y_test_cloud_patches[os_idx,:,:,:])
                count+=1
            X_test_cloud_patches = np.concatenate((X_test_cloud_patches,temp_x),axis=0)
            Y_test_cloud_patches = np.concatenate((Y_test_cloud_patches,temp_y),axis=0)







        print("test set size: ",X_test_cloud_patches.shape[0])
        print("test set size: ",Y_test_cloud_patches.shape[0])
        print("test samples added: ",num)

        fname_X_train = saved_patch_dir+"/X_train_patch_%01d.npy"%(i)
        fname_X_valid = saved_patch_dir+"/X_valid_patch_%01d.npy"%(i)
        fname_X_test  = saved_patch_dir+"/X_test_patch_%01d.npy"%(i)
        fname_Y_train = saved_patch_dir+"/Y_train_patch_%01d.npy"%(i)
        fname_Y_valid = saved_patch_dir+"/Y_valid_patch_%01d.npy"%(i)
        fname_Y_test  = saved_patch_dir+"/Y_test_patch_%01d.npy"%(i)

        np.save(fname_X_train,X_train_cloud_patches)
        np.save(fname_X_valid,X_valid_cloud_patches)
        np.save(fname_X_test,X_test_cloud_patches)
        np.save(fname_Y_train,Y_train_cloud_patches)
        np.save(fname_Y_valid,Y_valid_cloud_patches)
        np.save(fname_Y_test,Y_test_cloud_patches)
    
    print("Cloud Dataset Generated !")

if __name__=="__main__":
    print("Done!")