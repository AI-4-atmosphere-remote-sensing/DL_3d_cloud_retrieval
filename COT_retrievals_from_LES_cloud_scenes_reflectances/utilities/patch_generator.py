'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
'''
'''
Utility functions for generating patches from the image.
'''
# Import libraries
import numpy as np
import os

def extract_patch_from_image(image,kernel,stride=1):
    '''
    Generates patch from an image
    Args:
    Input: image: numpy image; kernel: array with patch height and width.
    e.g. of kernel is (10,10).
    Output:
    Patches: stored in patch_holder with dimension (#patches,kernel height, kernel width)
    
    Formula: ((W-KW)/SW)+1 W=width of img, KW=kernel width, SW= Stride width
    '''
    img_width = np.shape(image)[0]
    img_height = np.shape(image)[1]
    patch_height,patch_width = kernel

    r = np.int32(np.floor((img_height-patch_height)/stride))+1
    c = np.int32(np.floor((img_width-patch_width)/stride))+1

    patch_holder = np.empty((r*c,patch_height,patch_width),dtype=float) 
    i=0
    for row in range(r):
        for col in range(c):
            row_start = row*stride
            row_end = row_start+patch_height
            col_start = col*stride
            col_end = col_start+patch_width
            patch = image[row_start:row_end,col_start:col_end]
            patch_holder[i,:,:]=patch
            i+=1

    return patch_holder
        
def extract_patch(data,kernel,stride=1):
    '''
    Uses the extract_patch_from_image function to extract patches for images
    Args:
    Input:  1.data (numpy array). Dim:(samples,height,width,channels) e.g., dim = (102,72,72,2)
            2.kernel.Dim:(height,width) e.g., Value = (10,10)
            3.stride. int value.
    
    Output: patches (numpy array). Dim: (#patches,channels,kernel height, kernel width)
    '''
    total_samples    = np.shape(data)[0]
    sample_width     = np.shape(data)[1]
    sample_height    = np.shape(data)[2]
    channels         = np.shape(data)[3]
    patch_height,patch_width = kernel

    r = np.int32(np.floor((sample_height-patch_height)/stride))+1
    c = np.int32(np.floor((sample_width-patch_width)/stride))+1
    patch_holder = np.empty((r*c*total_samples,patch_height,patch_width,channels),dtype=float) 

    
    for k in range (channels):
        for i in range(total_samples):
            curr = extract_patch_from_image(np.squeeze(data[i,:,:,k]),kernel,stride)
            if i==0:
                prev = curr
            else:
                prev = np.concatenate((prev,curr),axis=0)
        patch_holder[:,:,:,k] = prev

    return patch_holder



if __name__=="__main__":
    patch_dir="/home/local/AD/ztushar1/Data/V12_cv_dataset_patches/"
    try:
        os.makedirs(patch_dir)
    except OSError as error:
        print("Directory Exists")

    processed_data_dir = "/home/local/AD/ztushar1/Data/V12_cv_dataset/"

    for i in range (5):
        fname_X_train = processed_data_dir+"X_train_%01d.npy"%(i)
        fname_X_valid = processed_data_dir+"X_valid_%01d.npy"%(i)
        fname_X_test  = processed_data_dir+"X_test_%01d.npy"%(i)
        fname_Y_train = processed_data_dir+"Y_train_%01d.npy"%(i)
        fname_Y_valid = processed_data_dir+"Y_valid_%01d.npy"%(i)
        fname_Y_test  = processed_data_dir+"Y_test_%01d.npy"%(i)

        X_train = np.load(fname_X_train)
        X_valid = np.load(fname_X_valid)
        X_test  = np.load(fname_X_test)
        Y_train = np.load(fname_Y_train)
        Y_valid = np.load(fname_Y_valid)
        Y_test  = np.load(fname_Y_test)

        X_train_patches = extract_patch(X_train,(10,10),stride=2)
        X_valid_patches = extract_patch(X_valid,(10,10),stride=2)
        X_test_patches  = extract_patch(X_test,(10,10),stride=2)
        Y_train_patches = extract_patch(Y_train,(10,10),stride=2)
        Y_valid_patches = extract_patch(Y_valid,(10,10),stride=2)
        Y_test_patches  = extract_patch(Y_test,(10,10),stride=2)

        print("x train size: ", np.shape(X_train_patches)[0])
        print("x valid size: ", np.shape(X_valid_patches)[0])
        print("x test size: ", np.shape(X_test_patches)[0])

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
    

    print("done !")