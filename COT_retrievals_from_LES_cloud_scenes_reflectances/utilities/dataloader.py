'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
'''

# Import Libraries
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import numpy as np
import torchvision.transforms as T
torch.manual_seed(0)

normalization_constant = dict()

normalization_constant['cv_dataset'] = {}
for fold in range(5):

    if fold==0:
        normalization_constant['cv_dataset'][fold]= [torch.tensor([0.1334, 0.0755], 
        dtype=torch.float64),torch.tensor([0.0199, 0.0079], dtype=torch.float64)]
    elif fold==1:
        normalization_constant['cv_dataset'][fold]= [torch.tensor([0.1454, 0.0807], 
        dtype=torch.float64),torch.tensor([0.0211, 0.0084], dtype=torch.float64)]

    elif fold==2:
        normalization_constant['cv_dataset'][fold]= [torch.tensor([0.1314, 0.0786], 
        dtype=torch.float64),torch.tensor([0.0189, 0.0077], dtype=torch.float64)]

    elif fold==3:
        normalization_constant['cv_dataset'][fold]= [torch.tensor([0.1270, 0.0749], 
        dtype=torch.float64),torch.tensor([0.0187, 0.0073], dtype=torch.float64)]

    elif fold==4:
        normalization_constant['cv_dataset'][fold]= [torch.tensor([0.1223, 0.0724], 
        dtype=torch.float64),torch.tensor([0.0189, 0.0077], dtype=torch.float64)]



normalization_constant['cloud_25'] = {}
for fold in range(5):

    if fold==0:
        normalization_constant['cloud_25'][fold]= [torch.tensor([ 0.1791,  0.1118, 10.5068], 
        dtype=torch.float64),torch.tensor([ 0.2290,  0.1144, 34.5334], dtype=torch.float64),
        torch.tensor([5.8406], dtype=torch.float64),
        torch.tensor([12.8047], dtype=torch.float64)]

    elif fold==1:
        normalization_constant['cloud_25'][fold]= [torch.tensor([ 0.1818,  0.1124, 10.8979], 
        dtype=torch.float64),torch.tensor([ 0.2325,  0.1150, 35.4430], dtype=torch.float64),
        torch.tensor([6.1521], dtype=torch.float64),
        torch.tensor([13.9813], dtype=torch.float64)]

    elif fold==2:
        normalization_constant['cloud_25'][fold]= [torch.tensor([ 0.1790,  0.1097, 11.0982], 
        dtype=torch.float64),torch.tensor([ 0.2359,  0.1158, 36.2888], dtype=torch.float64),
        torch.tensor([6.1969], dtype=torch.float64),
        torch.tensor([14.5031], dtype=torch.float64)]

    elif fold==3:
        normalization_constant['cloud_25'][fold]= [torch.tensor([0.1505, 0.0972, 9.1680], 
        dtype=torch.float64),torch.tensor([ 0.2199,  0.1088, 34.0264], dtype=torch.float64),
        torch.tensor([5.1292], dtype=torch.float64),
        torch.tensor([14.1287], dtype=torch.float64)]

    elif fold==4:
        normalization_constant['cloud_25'][fold]= [torch.tensor([0.1562, 0.1022, 8.9093], 
        dtype=torch.float64),torch.tensor([ 0.2152,  0.1091, 32.5128], dtype=torch.float64),
        torch.tensor([ 4.9177], dtype=torch.float64),
        torch.tensor([12.1297], dtype=torch.float64)]


normalization_constant['cloud_50'] = {}
for fold in range(5):

    if fold==0:
        normalization_constant['cloud_50'][fold]= [torch.tensor([0.2322, 0.1119], 
        dtype=torch.float64),torch.tensor([0.0255, 0.0099], dtype=torch.float64)]
    elif fold==1:
        normalization_constant['cloud_50'][fold]= [torch.tensor([0.2350, 0.1151], 
        dtype=torch.float64),torch.tensor([0.0258, 0.0100], dtype=torch.float64)]

    elif fold==2:
        normalization_constant['cloud_50'][fold]= [torch.tensor([0.2108, 0.1091], 
        dtype=torch.float64),torch.tensor([0.0236, 0.0094], dtype=torch.float64)]

    elif fold==3:
        normalization_constant['cloud_50'][fold]= [torch.tensor([0.2119, 0.1059], 
        dtype=torch.float64),torch.tensor([0.0239, 0.0092], dtype=torch.float64)]

    elif fold==4:
        normalization_constant['cloud_50'][fold]= [torch.tensor([0.2218, 0.1096], 
        dtype=torch.float64),torch.tensor([0.0254, 0.0102], dtype=torch.float64)]


normalization_constant['cloud_75'] = {}
for fold in range(5):

    if fold==0:
        normalization_constant['cloud_75'][fold]= [torch.tensor([0.2470, 0.1166], 
        dtype=torch.float64),torch.tensor([0.0260, 0.0099], dtype=torch.float64)]
    elif fold==1:
        normalization_constant['cloud_75'][fold]= [torch.tensor([0.2464, 0.1186], 
        dtype=torch.float64),torch.tensor([0.0257, 0.0097], dtype=torch.float64)]

    elif fold==2:
        normalization_constant['cloud_75'][fold]= [torch.tensor([0.2214, 0.1123], 
        dtype=torch.float64),torch.tensor([0.0234, 0.0091], dtype=torch.float64)]

    elif fold==3:
        normalization_constant['cloud_75'][fold]= [torch.tensor([0.2259, 0.1110], 
        dtype=torch.float64),torch.tensor([0.0242, 0.0090], dtype=torch.float64)]

    elif fold==4:
        normalization_constant['cloud_75'][fold]= [torch.tensor([0.2419, 0.1169], 
        dtype=torch.float64),torch.tensor([0.0263, 0.0103], dtype=torch.float64)]

class NasaDataset(Dataset):
    """  Dataset types:
        1. 'cloud_25'
        2. 'cloud_50'
        3. 'cloud_75'
        4. 'cv_dataset'
        """

    def __init__(self, dataset_name, fold, X, Y, cropped=False, transform=False):
        """
        X = reflactance data (batch,10,10,2)
        Y = cloud optical thickness data (batch,10,10,2)
        cropped  = default is False. Make true for okamura model
        """
        self.fold = fold
        self.cropped   = cropped
        self.transform = transform
        self.dataset_name = dataset_name


        # New script loads directly the patch image
        self.r_data   = X
        self.cot_data = Y 
       
        self.transform1 = T.Compose([T.ToTensor()])
        if self.transform:
            mean1, std1, mean2, std2     = self.get_mean_std()
            self.normalize_in  = T.Compose([T.Normalize(mean1, std1)])
            self.normalize_out = T.Compose([T.Normalize(mean2, std2)])
        
        # For baseline model, the output has a shape of 6x6. Hence this transform is used.
        self.crop_func1 = torch.nn.Sequential(T.CenterCrop(6))
        self.crop_func2 = torch.nn.Sequential(T.CenterCrop(8))

    def __len__(self):
        return self.r_data.shape[0]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        reflectance = self.r_data[idx,:,:,:]
        cot = self.cot_data[idx, :,:,:]
        reflectance = self.transform1(reflectance)
        cot  = self.transform1(cot)

        if self.transform:
            reflectance = self.normalize_in(reflectance)
            # cot[0,:,:] = self.normalize_out(torch.unsqueeze(cot[0,:,:],0))
            cot[0,:,:] = torch.log(torch.unsqueeze(cot[0,:,:],0)+0.01)

        if self.cropped==1:
            cot = self.crop_func1(cot)
        elif self.cropped==2:
            cot = self.crop_func2(cot)
        sample = {'reflectance': reflectance, 'cot': cot}
        return sample

    def get_mean_std(self):
        mean1, std1, mean2, std2 = normalization_constant[self.dataset_name][self.fold]  
        return mean1, std1, mean2, std2

    def get_max(self):
        print("COT MAX",np.max(self.cot_data[:,:,:,0]))
        print("COT MIN",np.min(self.cot_data[:,:,:,0]))
        print("Reflectance MAX",np.max(self.r_data[:,:,:,0]))
        print("Reflectance MIN",np.min(self.r_data[:,:,:,0]))
    
    def get_full_data(self):
        return self.r_data,self.cot_data

if __name__=="__main__":
    # Dataset Directory

    data_dir="preprocessed_data/Cloud_25"
    for i in range (5):

        fname_X_train = data_dir+"/X_train_patch_%01d.npy"%(i)
        fname_X_valid = data_dir+"/X_valid_patch_%01d.npy"%(i)
        fname_X_test  = data_dir+"/X_test_patch_%01d.npy"%(i)
        fname_Y_train = data_dir+"/Y_train_patch_%01d.npy"%(i)
        fname_Y_valid = data_dir+"/Y_valid_patch_%01d.npy"%(i)
        fname_Y_test  = data_dir+"/Y_test_patch_%01d.npy"%(i)

        X_train = np.load(fname_X_train)
        X_valid = np.load(fname_X_valid)
        X_test  = np.load(fname_X_test)
        Y_train = np.load(fname_Y_train)
        Y_valid = np.load(fname_Y_valid)
        Y_test  = np.load(fname_Y_test)

        if i==4:
            break

    train_data = NasaDataset(dataset_name ='cloud_25', fold = i, X=X_train, Y= Y_train,cropped=False, transform=False)
    loader = DataLoader(train_data, batch_size=10)
    train_data.get_max()
    print(train_data.get_mean_std())



    test_data = NasaDataset(dataset_name ='cloud_25', fold = i, X=X_valid, Y= Y_valid,cropped=False, transform=False)
    loader = DataLoader(test_data, batch_size=10)
    sample = loader.dataset[100]
    print(len(test_data))
    print(sample['cot'].shape)
    print(sample['reflectance'].shape)
    test_data.get_max()
    
    temp1 = []
    temp2 = []
    for i in range(len(loader.dataset)):
        data = loader.dataset[i]
        # get the data
        X, Y = data['reflectance'][0:2,:,:],data['cot'][0,:,:]
        # print(torch.max(Y))
        temp1.append(torch.max(Y))
        temp2.append(torch.min(Y))
        
    print(max(temp1))
    print(min(temp2))
    
    # to compute
    print("Done !")
