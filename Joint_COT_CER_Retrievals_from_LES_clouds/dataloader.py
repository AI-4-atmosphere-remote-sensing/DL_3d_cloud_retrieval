'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
'''


# Import Libraries
import os
import h5py
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import numpy as np
import torchvision.transforms as T
torch.manual_seed(0)

normalization_constant = dict()

normalization_constant['cv_dataset'] = {}
for fold in range(5):

    if fold==0:
        normalization_constant['cv_dataset'][fold]= [torch.tensor([0.1177, 0.0970], 
        dtype=torch.float64),torch.tensor([0.1906, 0.1267], dtype=torch.float64)]
    elif fold==1:
        normalization_constant['cv_dataset'][fold]= [torch.tensor([0.1178, 0.0967], 
        dtype=torch.float64),torch.tensor([0.1909, 0.1264], dtype=torch.float64)]

    elif fold==2:
        normalization_constant['cv_dataset'][fold]= [torch.tensor([0.1176, 0.0965], 
        dtype=torch.float64),torch.tensor([0.1906, 0.1262], dtype=torch.float64)]

    elif fold==3:
        normalization_constant['cv_dataset'][fold]= [torch.tensor([0.1171, 0.0960], 
        dtype=torch.float64),torch.tensor([0.1909, 0.1257], dtype=torch.float64)]

    elif fold==4:
        normalization_constant['cv_dataset'][fold]= [torch.tensor([0.1170, 0.0963], 
        dtype=torch.float64),torch.tensor([0.1903, 0.1261], dtype=torch.float64)]


class NasaDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading and processing data related to LES (Large Eddy Simulation) profiles.

    Parameters:
        fold (int): Specifies the current fold for cross-validation, chooses the correct normalization constants.
        profilelist (list): A list of profile identifiers to load specific data samples/ LES profiles.
        root_dir1 (str): Path to the primary directory containing data.
        root_dir2 (str): Path to the secondary directory containing additional data (if used).
        patch_size (int): The size of the patches to extract from the data (default: 64).
        stride (int): The stride value for patch extraction (default: 10).
        cp (bool/ int): Flag indicating whether cropping is enabled (default: False).Options: 1,2
        transform (bool, optional): Flag indicating whether data transformations are applied (default: True).
    """

    def __init__(self, fold=None,profilelist=None, root_dir1=None, root_dir2=None,patch_size=64,stride=2,cp=False,transform=True ):
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.profilelist = profilelist
        self.patch_size = patch_size
        self.stride = stride
        # self.filelist = os.listdir(root_dir)
        self.transform1 = T.Compose([T.ToTensor()])
        self.transform =transform
        if self.transform:
            mean1, std1     = self.get_mean_std(fold)
            self.transform2 = T.Compose([T.Normalize(mean1, std1)])

        self.crop_func1 = torch.nn.Sequential(T.CenterCrop(6))
        self.crop_func2 = torch.nn.Sequential(T.CenterCrop(60))
        self.cp=cp

    def get_mean_std(self,fold):
        mean1, std1= normalization_constant['cv_dataset'][fold]  
        return mean1, std1

    def __len__(self):
        return len(self.profilelist)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        p_num = self.profilelist[idx]
        if p_num>800:
            fname = self.root_dir2+"/LES_profile_%05d.hdf5"%(p_num-1000)
        else:
            fname = self.root_dir1+"/LES_profile_%05d.hdf5"%(p_num)
        hf = h5py.File(fname, 'r')
        # print(hf.keys()) 

        r_data = np.empty((144,144,2), dtype=float) 
        # cmask =np.empty((144,144),dtype=float) 
        temp              = np.nan_to_num(np.array(hf.get("Reflectance_100m_resolution")))
        # reflectance at 0.66 um
        r_data[:,:,0]   = temp[0,:,:]
        # r_data[:,:,1]   = temp[0,:,:]
        # r_data[:,:,2]   = temp[0,:,:]
        # # reflectance at 2.13 um
        r_data[:,:,1]   = temp[1,:,:]
        # print(r_data[10,10,0])

        # # cot profile
        cot_data =np.empty((144,144,2),dtype=float) 
        cot_data[:,:,0] = np.nan_to_num(np.array(hf.get("Cloud_optical_thickness_(100m resolution)")))

        # # CER
        cot_data[:,:,1] = np.nan_to_num(np.array(hf.get("CER_(100m resolution)")))

        # # SZA and VZA
        # sza = np.nan_to_num(np.array(hf.get("SZA"))) 
        # vza = np.nan_to_num(np.array(hf.get("VZA"))) 
        
        # Cloud Mask
        # r_data[:,:,2] = np.nan_to_num(np.array(hf.get("Native_Cloud_fraction_(100m)")))    
        # cot_data[:,:,2] =  np.nan_to_num(np.array(hf.get("Native_Cloud_fraction_(100m)")))    
        hf.close()
        

        # Transformation
        cot_data[:,:,0] = np.log(cot_data[:,:,0]+1)
        cot_data[:,:,1] = cot_data[:,:,1] /30.0

        # Extract patches
        patches_r      = self.extract_patches(r_data)
        # patch_labels = self.extract_patches(np.expand_dims(cmask, axis=2))
        patch_cots    = self.extract_patches(cot_data)

        # Convert to tensor
        if self.transform1:
            patches_r = [self.transform1(patch) for patch in patches_r]   
            patch_cots = [self.transform1(patch) for patch in patch_cots]  
        if self.transform:
            patches_r = [self.transform2(patch) for patch in patches_r]        
        if self.cp==1:
            patch_cots = [self.crop_func1(patch) for patch in patch_cots] 
        elif self.cp==2:
            patch_cots = [self.crop_func2(patch) for patch in patch_cots] 
        # sample = {'patches':patches,'labels':patch_labels,'cot':patch_cots}
        sample = {'rad_patches':patches_r,'cot_patches':patch_cots}
        return sample


    def extract_patches(self, image):
        h, w, c = image.shape
        patches = []

        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                patch = image[i:i + self.patch_size, j:j + self.patch_size]
                patches.append(patch)

        return patches
    

if __name__=="__main__":


    def reconstruct_image(patches, original_shape, patch_size, stride):
        # Get the height, width, and number of channels of the original image
        h, w, c = original_shape
        # Initialize an array to store the reconstructed image
        reconstructed_image = np.zeros(original_shape)

        # Initialize counters for patch and position in the reconstructed image
        patch_index = 0

        # Iterate through the image using a sliding window approach
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                # Place the current patch in the corresponding position in the reconstructed image
                reconstructed_image[i:i + patch_size, j:j + patch_size] = patches[patch_index]
                # Move to the next patch
                patch_index += 1

        return reconstructed_image

    for fold in range(1):    
        # fold=0
        print("checking fold: ", fold)
        dataset_dir1 = "one_thousand_profiles/Refl"
        dataset_dir2 = "ncer_fill3"
        cv_train_list = np.load("Data_split/train_split_100m.npy")
        cv_valid_list = np.load("Data_split/valid_split_100m.npy")
        cv_test_list = np.load("Data_split/test_split_100m.npy")
        profilelist  = cv_train_list[fold][:,0]

        train_data = NasaDataset(fold=fold,profilelist=profilelist,
                                root_dir1=dataset_dir1,root_dir2=dataset_dir2,
                                patch_size=64,stride=6, cp=2)
        print(len(train_data))
        loader = DataLoader(train_data, batch_size=10,shuffle=False)
        
        temp= []
        temp1= []   
        for i in range(len(loader.dataset)):
            data = loader.dataset[i]
            # get the data
            X, Z= data['rad_patches'],data['cot_patches']
            # print(len(X),len(Z))

            print(X[0].shape)
            print(Z[0].shape)


            # rr_cot = reconstruct_image(Z, (144,144,2), 6, 6)
            # # Check if the arrays are equal
            # are_equal = np.array_equal(rr_cot, real_cot)

            # # Print the result
            # print("Arrays are equal:", are_equal)
            break
        # break  
    #     el = [torch.max(patch[1,:,:]).item() for patch in Z]  
    #     em = [torch.min(patch[1,:,:]).item() for patch in Z] 
    #     temp.append(np.max(el))
    #     temp1.append(np.min(em))
    # print(np.max(temp),np.min(temp1))
    # dataset_dir2 = "/home/local/AD/ztushar1/Data/ncer_fill3"

    # test_data = NasaDataset(root_dir=dataset_dir2)
    # loader = DataLoader(test_data, batch_size=10,shuffle=False)
    # print(len(loader.dataset))
    # for i in range(len(loader.dataset)):
    #     data = loader.dataset[i]
    #     # get the data
    #     X, Y, Z = data['reflectance'],data['cmask'],data['patches']
    #     print(Y.shape, X.shape,Z[0].shape)
    #     break  
    
