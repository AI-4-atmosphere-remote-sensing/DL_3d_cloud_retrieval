'''
   Author: Zahid Hassan Tushar
   Email: ztushar1@umbc.edu

'''
# Import Libraries
from fileinput import filename
import numpy as np
import os
import torch
import csv
import argparse
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from model_config import CloudUNet, DNN2w
from utilities import get_pred, plot_cot, plot_cmask

normalization_constant = dict()
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

def parse_args():
    parser = argparse.ArgumentParser(description='Test a model')
    parser.add_argument('--model_arch', type=str,  default='cloudunet', help='Model Architecture')
    parser.add_argument('--model_filename', type=str,  default='cloudunet.pth', help='Model Architecture')

    args = parser.parse_args()
    return args



def get_mean_std(dataset_name, fold):
    mean1, std1, mean2, std2 = normalization_constant[dataset_name][fold]  
    return mean1, std1, mean2, std2

def get_profile_pred1(model,model_arch,X_test,Y_test,dataset_name,fold,stride):

    #1 Define the mechanism for patch extraction
    map = np.zeros_like(np.squeeze(Y_test))
    Y_pred = np.zeros_like(map)

    kernel = (10,10)

    img_width = X_test.shape[0]
    img_height = X_test.shape[1]
    patch_height,patch_width = kernel

    r = np.int32(np.ceil((img_height-patch_height)/stride))+1
    c = np.int32(np.ceil((img_width-patch_width)/stride))+1

    test_losses = []
    test_losses2 = [] # to store loss at 6x6 patch size


    #2 convert to tensor
    X_test = TF.to_tensor(X_test)
    Y_test = TF.to_tensor(Y_test)

    #3 Normalize data
    mean1, std1, mean2, std2     = get_mean_std(dataset_name,fold)
    normalize_in  = T.Compose([T.Normalize(mean1, std1)])
    # normalize_out = T.Compose([T.Normalize(mean2, std2)])
    X_test = normalize_in(X_test)
    Y_test = torch.log(Y_test+0.01)
    # Y_test = normalize_out(Y_test)

    predictions  = []
    ground_truth = []


    for row in range(r):
        for col in range(c):
            row_start = min(row*stride,img_height-patch_height)
            row_end = row_start+patch_height
            col_start =  min(col*stride,img_width-patch_width)
            col_end = col_start+patch_width
            patch = X_test[0:2,row_start:row_end,col_start:col_end]
            if model_arch=="okamura":
                label = Y_test[0,row_start+2:row_end-2,col_start+2:col_end-2]
            else:
                label = Y_test[0,row_start:row_end,col_start:col_end]

            

            test_loss,patch_pred = get_pred(model=model,X_test=patch,Y_test=label)
            test_losses.append(test_loss)
            
            # Store predicted patch and ground truth patch
            predictions.append(patch_pred)
            ground_truth.append(label.cpu().detach().numpy())


            if model_arch=="cloudunet":
                label2 = label.cpu().detach().numpy()[2:-2,2:-2]
                patch_pred2 = patch_pred[2:-2,2:-2]
                test_losses2.append((np.square(label2 - patch_pred2)).mean())
            else:
                test_losses2=test_losses
            
            
    # print("Test Loss: ",np.average(test_losses))
    return ground_truth,predictions,test_losses,test_losses2


def get_profile_pred2(model,model_arch,X_test,Y_test,dataset_name,fold,stride):

    #1 Define the mechanism for patch extraction
    map = np.zeros_like(np.squeeze(Y_test))
    Y_pred = np.zeros_like(map)

    kernel = (10,10)

    img_width = X_test.shape[0]
    img_height = X_test.shape[1]
    patch_height,patch_width = kernel

    r = np.int32(np.ceil((img_height-patch_height)/stride))+1
    c = np.int32(np.ceil((img_width-patch_width)/stride))+1

    test_losses = []


    #2 convert to tensor
    X_test = TF.to_tensor(X_test)
    Y_test = TF.to_tensor(Y_test)

    #3 Normalize data
    mean1, std1, mean2, std2     = get_mean_std(dataset_name,fold)
    normalize_in  = T.Compose([T.Normalize(mean1, std1)])
    # normalize_out = T.Compose([T.Normalize(mean2, std2)])
    X_test = normalize_in(X_test)
    Y_test = torch.log(Y_test+0.01)
    # Y_test = normalize_out(Y_test)

    # patch_holder = np.empty((r*c,patch_height,patch_width),dtype=float) 
    for row in range(r):
        for col in range(c):
            row_start = min(row*stride,img_height-patch_height)
            row_end = row_start+patch_height
            col_start =  min(col*stride,img_width-patch_width)
            col_end = col_start+patch_width
            patch = X_test[0:2,row_start:row_end,col_start:col_end]
            if model_arch=="okamura":
                label = Y_test[0,row_start+2:row_end-2,col_start+2:col_end-2]
            elif model_arch=="okamura2":
                label = Y_test[0,row_start+1:row_end-1,col_start+1:col_end-1]
            else:
                label = Y_test[0,row_start:row_end,col_start:col_end]



            test_loss,patch_pred = get_pred(model=model,X_test=patch,Y_test=label)
            test_losses.append(test_loss)
            # map[row_start:row_end,col_start:col_end] =map[row_start:row_end,col_start:col_end] +np.ones((10,10))
            if model_arch=="okamura":
                map[row_start+2:row_end-2,col_start+2:col_end-2] +=1
                Y_pred[row_start+2:row_end-2,col_start+2:col_end-2] +=patch_pred  
            elif model_arch=="okamura2":
                map[row_start+1:row_end-1,col_start+1:col_end-1] +=1
                Y_pred[row_start+1:row_end-1,col_start+1:col_end-1] +=patch_pred               
            else:            
                map[row_start:row_end,col_start:col_end] +=1
                Y_pred[row_start:row_end,col_start:col_end] +=patch_pred
    
    if model_arch=="okamura":
        Y_pred = Y_pred[2:-2,2:-2]/map[2:-2,2:-2]
        Y_test = Y_test[0,2:-2,2:-2]
    elif model_arch=="okamura2":
        Y_pred = Y_pred[1:-1,1:-1]/map[1:-1,1:-1]
        Y_test = Y_test[:,1:-1,1:-1]
    else:
        
        Y_pred=Y_pred/map
        Y_test = Y_test[0,:,:]

    print("Test Loss: ",np.average(test_losses))
    Y_test = Y_test.cpu().detach().numpy()
    return Y_test, Y_pred, (np.square(Y_test - Y_pred)).mean()



if __name__=="__main__":
    # Parse the arguments
    args = parse_args()
    model_arch = args.model_arch
    model_filename = args.model_filename



    # Load Test Data
    data_dir = "test_data/"
    dataset_name = "cloud_25"

    fname_X_test = data_dir+"X_test.npy"
    fname_Y_test = data_dir+"Y_test.npy"
    X_test  = np.load(fname_X_test)
    Y_test  = np.load(fname_Y_test)


    # Load saved model
    # model_arch = "cloudunet"
    # model_filename = "cloudunet.pth"
    if model_arch == "cloudunet":
        model = CloudUNet(n_channels=2,n_classes=1)
        filename = "saved_model/"+model_arch+"/"+model_filename
        model.load_state_dict(torch.load(filename))
        stride = 10
        header = ['Name','Loss (log)','Loss (log) 6x6']

    elif model_arch == "okamura":
        model = DNN2w()
        filename = "saved_model/"+model_arch+"/"+model_filename
        model.load_state_dict(torch.load(filename))
        stride = 6
        header = ['Name','Loss (log)','Loss (log) 6x6']

    # Generate Predictions Apple2Apple [Patch]
    result_path = os.path.join("predictions",model_arch)
    try:
        os.makedirs(result_path)
    except OSError as error:
        print("Directory Exists")

    losscsv = "/loss_log_patches_stride_%02d.csv"%(10) # to store loss for each patch

    # Plotting Properties
    use_log=False
    limit1 = [-6.6226,9.3373]

    csv_name = result_path+losscsv
    with open(csv_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)

        for l in range(X_test.shape[0]):  
            ground_truth, predictions,patch_losses,patch_losses2 = get_profile_pred1(model=model,model_arch=model_arch,
            X_test=X_test[l,:,:,:],Y_test=Y_test[l,:,:,0],dataset_name=dataset_name,
            fold = 4, stride=10)

            p_num      = np.int32(Y_test[l,1,1,2]) # profile number


            for patch_idx in range(len(predictions)):
                if model_arch=="cloudunet":
                    gt = ground_truth[patch_idx][2:-2,2:-2]
                    pred = predictions[patch_idx][2:-2,2:-2]
                    patch_filename = "profile_%01d_patch_%02d"%(p_num,patch_idx)
                    writer.writerow([patch_filename,patch_losses[patch_idx],patch_losses2[patch_idx]])
                else:
                    gt = ground_truth[patch_idx]
                    pred = predictions[patch_idx]
                    patch_filename = "profile_%01d_patch_%02d"%(p_num,patch_idx)
                    writer.writerow([patch_filename,patch_losses[patch_idx]])

                fname = result_path+"/apple2apple_jet_norm_cot_profile_%01d_patch_%02d.png"%(p_num,patch_idx)
                plot_cot(cot=gt,title="COT",fname=fname,use_log=use_log,limit=limit1)

                fname = result_path+"/apple2apple_jet_norm_pred_profile_%01d_patch_%02d.png"%(p_num,patch_idx)
                plot_cot(cot=pred,title="Pred",fname=fname,use_log=use_log,limit=limit1)

                fname = result_path+"/apple2apple_jet_norm_cmask_profile_%01d_patch_%02d.png"%(p_num,patch_idx)
                plot_cmask(cmask=Y_test[l,2:-2,2:-2,1],title="Cmask",fname=fname)
            
            print("Patch Predictions generated for Profile: ",p_num)


    losscsv2 = "/loss_log_profiles_stride_%02d.csv"%stride # to store loss for each patch
    csv_name = result_path+losscsv

    with open(csv_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)    
        writer.writerow(header)

        for l in range(X_test.shape[0]):  
            profile, pred, loss = get_profile_pred2(model=model,model_arch=model_arch,
            X_test=X_test[l,:,:,:],Y_test=Y_test[l,:,:,0],dataset_name=dataset_name,
            fold = 4, stride=stride)

            p_num      = np.int32(Y_test[l,1,1,2]) # Profile Number


            profile_filename = "full_profile_jet_norm_IPA_%01d"%(p_num)

            writer.writerow([profile_filename,loss])

            use_log=False
            limit1 = [-6.6226,9.3373]

            fname = result_path+"/full_profile_jet_norm_cot_%01d.png"%(p_num)
            plot_cot(cot=profile,title="COT",fname=fname,use_log=use_log,limit=limit1)

            fname = result_path+"/full_profile_jet_norm_pred_%01d.png"%(p_num)
            plot_cot(cot=pred,title="Pred",fname=fname,use_log=use_log,limit=limit1)

            fname = result_path+"/full_profile_jet_norm_cmask_%01d.png"%(p_num)
            plot_cmask(cmask=Y_test[l,2:-2,2:-2,1],title="Cmask",fname=fname)

            print("Full Profile Predictions generated for Profile: ",p_num)