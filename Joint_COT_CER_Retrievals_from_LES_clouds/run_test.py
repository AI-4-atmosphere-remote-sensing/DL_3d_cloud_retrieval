'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
'''

'''get_predictions
Test the performance of the model using the profiles. 
1. best or worst Performance based on MSE/avg of COT values.
2. outcast: 100% cloudy pixels
Usage: Comment out the #model_name and #filename
'''

import numpy as np
import torch
import csv
import os
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim
import h5py

from metrics import *
from utilities import get_predictions
from visualization import *

# import Models
from cam import CAM


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

def get_mean_std(fold):
    mean1, std1= normalization_constant['cv_dataset'][fold]  
    return mean1, std1


def get_profile_pred(model,model_name,X_test,Y_test,patch_size,stride):
    # stride = 2
    img_width = X_test.shape[1]
    img_height = X_test.shape[0]
    patch_height,patch_width = patch_size,patch_size

    r = np.int32(np.ceil((img_height-patch_height)/stride))+1
    c = np.int32(np.ceil((img_width-patch_width)/stride))+1

    #2 convert to tensor    
    X_test = TF.to_tensor(X_test)
    Y_test = TF.to_tensor(Y_test)

    #3 Normalize data
    mean1, std1     = get_mean_std(fold)
    transform2 = T.Compose([T.Normalize(mean1, std1)])
    X_test  = transform2(X_test)
    Y_test[0,:,:] = torch.log(Y_test[0,:,:]+1)
    Y_test[1,:,:] = Y_test[1,:,:]/30.0

    # initialize mapping array
    map = np.zeros_like(np.squeeze(Y_test))
    Y_pred = np.zeros_like(map)
    for row in range(r):
        for col in range(c):
            row_start = min(row*stride,img_height-patch_height)
            row_end = row_start+patch_height
            col_start =  min(col*stride,img_width-patch_width)
            col_end = col_start+patch_width
            patch = X_test[:,row_start:row_end,col_start:col_end]
            patch_pred = get_predictions(model=model,X_test=patch,Y_test=None)

            if model_name=="okamura":
                map[:,row_start+2:row_end-2,col_start+2:col_end-2] +=1
                Y_pred[:,row_start+2:row_end-2,col_start+2:col_end-2] +=patch_pred  
            elif model_name=="okamura2":
                map[:,row_start+2:row_end-2,col_start+2:col_end-2] +=1
                Y_pred[:,row_start+2:row_end-2,col_start+2:col_end-2] +=patch_pred              
            else:            
                map[:,row_start:row_end,col_start:col_end] +=1
                Y_pred[:,row_start:row_end,col_start:col_end] +=patch_pred

    

    Y_pred = Y_pred[:,2:-2,2:-2]/map[:,2:-2,2:-2]
    Y_test = Y_test[:,2:-2,2:-2]


    Y_test = Y_test.cpu().detach().numpy()

    cot_mse_loss = compute_mse(Y_pred[0,:,:],Y_test[0,:,:])
    cer_mse_loss = compute_mse(Y_pred[1,:,:]*30,Y_test[1,:,:]*30)

    cot_abs_error = np.abs((Y_test[0,:,:] - Y_pred[0,:,:])).mean()
    cer_abs_error = np.abs((Y_test[1,:,:]*30 - Y_pred[1,:,:]*30)).mean()

    cot_rc = compute_rc(Y_pred[0,:,:],Y_test[0,:,:])
    cer_rc = compute_rc(Y_pred[1,:,:]*30,Y_test[1,:,:]*30)

    output_scores = {"cot_mse_loss":cot_mse_loss,   "cer_mse_loss":cer_mse_loss,
                     "cot_mae_loss":cot_abs_error, "cer_mae_loss":cer_abs_error,
                     "cot_rc":cot_rc, "cer_rc":cer_rc}

    return Y_test, Y_pred, output_scores


def run_test(out,stride=4):
    #extract parameters
    saved_model_dir = out['saved_model_dir']
    model_name      = out['model_name']
    model_filenames = out['model_filenames']
    logfile         = out['logfile']
    patch_size      = out['patch_size']
    
    # Initialize the model
    cp=False
    if model_name =="cam":
        model =CAM(in_channels=2,gate_channels=64)
     
    # dataset dir
    dataset_dir1 = "/home/local/AD/ztushar1/COT_CER_Joint_Retrievals/one_thousand_profiles/Refl"
    dataset_dir2 = "/home/local/AD/ztushar1/COT_CER_Joint_Retrievals/ncer_fill3"
    cv_test_list = np.load("/home/local/AD/ztushar1/COT_CER_Joint_Retrievals/Data_split/test_split_100m.npy")


    # create a directory and a csv file to store the scores
    dir_name = saved_model_dir+"/Visuals_"+logfile+"/Stride_"+str(stride)
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        print("folder already exists")

    header = ["Name","MSE","MAE","RC"]
    losscsv = "/loss_log_stride_%01d.csv"%(stride)

    csv_name = dir_name+losscsv
    with open(csv_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)    
        writer.writerow(header)    

        temp2,  temp4,  temp6  = [],[],[]
        temp12, temp14, temp16 = [],[],[]
        total_models = len(model_filenames)

        # For each fold, load data and model and run inference
        for var in range (total_models):
            if total_models==1:
                i=4
                filename        = saved_model_dir + "/"+model_filenames[0]
            else:
                i=var
                filename        = saved_model_dir + "/"+model_filenames[i]


            # load model
            model.load_state_dict(torch.load(filename,map_location=torch.device('cpu')))

            # empty list to hold different metric scores
            temp1,  temp3,  temp5   = [],[],[]
            temp11, temp13, temp15  = [],[],[]

            cot_best=[]
            cer_best=[]     
  
            profilelist =  cv_test_list[i][:,0]
            # for each profile compute the scores
            for l in profilelist:
                p_num= np.int16(l) 

                if p_num>800:
                    fname = dataset_dir2+"/LES_profile_%05d.hdf5"%(p_num-1000)
                else:
                    fname = dataset_dir1+"/LES_profile_%05d.hdf5"%(p_num)
                hf = h5py.File(fname, 'r')
                # initialize the array
                X_test = np.empty((144,144,2), dtype=float)
                Y_test = np.empty((144,144,2),dtype=float) 
                temp              = np.nan_to_num(np.array(hf.get("Reflectance_100m_resolution")))
                # reflectance at 0.66 um
                X_test[:,:,0]   = temp[0,:,:]
                # reflectance at 2.13 um
                X_test[:,:,1]   = temp[1,:,:]
                # cot profile
                Y_test[:,:,0]   = np.nan_to_num(np.array(hf.get("Cloud_optical_thickness_(100m resolution)")))

                # CER
                Y_test[:,:,1] = np.nan_to_num(np.array(hf.get("CER_(100m resolution)")))
                hf.close()

                profile, pred, output_scores = get_profile_pred(model=model,model_name=model_name,
                X_test=X_test,Y_test=Y_test,patch_size=patch_size,stride=stride)


                ##################### Multiply by 30 ##################################
                profile[1,:,:] = profile[1,:,:]*30
                pred[1,:,:]    = pred[1,:,:]*30

                # temp = (pred[0,:,:]>=0)*1
                # pred[0,:,:]   = pred[0,:,:]*temp

                # store COT losses
                temp1.append(output_scores['cot_mse_loss'])
                temp3.append(output_scores['cot_mae_loss'])             
                temp5.append(output_scores['cot_rc'])               


                # store CER losses
                temp11.append(output_scores['cer_mse_loss'])
                temp13.append(output_scores['cer_mae_loss'])
                temp15.append(output_scores['cer_rc'])


                if out['flag_plot'] and i==4: # Generate visuals only for the last model

                    profile_filename = "full_profile_jet_norm_IPA_%01d"%(p_num)
                    writer.writerow([profile_filename,output_scores['cot_mse_loss'],output_scores['cot_mae_loss'], output_scores["cot_rc"]])
                    writer.writerow([profile_filename,output_scores['cer_mse_loss'],output_scores['cer_mae_loss'], output_scores["cer_rc"]])
                
                    # Plot only for final fold
                    use_log=False
                    # limit1 = [-6.6226,9.3373]
                    limit1 = [0,7]
                    limit2 = [0,40]

                    if p_num in [579,1093,580,1095,1075]:

                        # Plot COT
                        fname = dir_name+"/full_profile_jet_cot_%01d.png"%(p_num)
                        plot_cot2(cot=profile[0,:,:],title="COT",fname=fname,use_log=use_log,limit=limit1)

                        fname = dir_name+"/full_profile_jet_pred_cot_%01d.png"%(p_num)
                        plot_cot2(cot=pred[0,:,:],title="Retrieved COT",fname=fname,use_log=use_log,limit=limit1)

                        fname = dir_name+"/full_profile_jet_abs_error_cot_%01d.png"%(p_num)
                        plot_cot2(cot=np.abs(profile[0,:,:]-pred[0,:,:]),title="Absolute Error COT",fname=fname,use_log=use_log,limit=limit1)     

                        # Plot CER
                        fname = dir_name+"/full_profile_jet_norm_cer_%01d.png"%(p_num)
                        plot_cot2(cot=profile[1,:,:],title="CER",fname=fname,use_log=use_log,limit=limit2)

                        fname = dir_name+"/full_profile_jet_norm_pred_cer_%01d.png"%(p_num)
                        plot_cot2(cot=pred[1,:,:],title="Retrieved CER",fname=fname,use_log=use_log,limit=limit2)

                        fname = dir_name+"/full_profile_jet_norm_abs_error_cer_%01d.png"%(p_num)
                        plot_cot2(cot=np.abs(profile[1,:,:]-pred[1,:,:]),title="Absolute Error CER",fname=fname,use_log=use_log,limit=limit2) 
                        


            temp2.append(np.average(temp1))
            temp4.append(np.average(temp3))
            temp6.append(np.average(temp5))

            temp12.append(np.average(temp11))
            temp14.append(np.average(temp13))
            temp16.append(np.average(temp15))

            if total_models==1:
                break      


        writer.writerow([filename,"COT MSE",np.average(temp2),np.std(temp2)])
        writer.writerow([filename,"COT MAE",np.average(temp4),np.std(temp4)])
        writer.writerow([filename,"COT RC",np.average(temp6),np.std(temp6)])


        writer.writerow([filename,"CER MSE",np.average(temp12),np.std(temp12),0])
        writer.writerow([filename,"CER MAE",np.average(temp14),np.std(temp14),0])
        writer.writerow([filename,"CER RC",np.average(temp16),np.std(temp16),0])

        # print the cot loss
        print("COT MSE: ",       np.average(temp2),np.std(temp2))
        print("COT MAE: ",      np.average(temp4),np.std(temp4))
        print("COT RC: ",      np.average(temp6),np.std(temp6))

        # print the cer loss
        print("CER MSE: ",       np.average(temp12),np.std(temp12))
        print("CER MAE: ",      np.average(temp14),np.std(temp14))
        print("CER RC: ",      np.average(temp16),np.std(temp16))


if __name__=="__main__":
    saved_model_dir = "saved_model/"
    patch_size = 64
    stride = 10
    model_name  = "cam"
    model_filenames = [
                    "cam_fold_0.pth",
                    "cam_fold_1.pth",
                    "cam_fold_2.pth",
                    "cam_fold_3.pth",
                    "cam_fold_4.pth"]  

    out = {"model_name": model_name,"saved_model_dir":saved_model_dir+model_name,
            "model_filenames":model_filenames, "patch_size":patch_size,
           "logfile":"Results", "flag_plot":False}
    run_test(out,stride)
    
    print("Done!")
