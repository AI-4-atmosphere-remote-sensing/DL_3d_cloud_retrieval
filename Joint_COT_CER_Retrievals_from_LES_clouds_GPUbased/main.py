
'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
'''
'''
This is the main file for training in Ada. It jointly trains for CER and COT Retrievals
'''

# Import libraries
import timeit

start = timeit.default_timer()

# import libraries
import argparse
import os
from torchinfo import summary
import time
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)

from dataloader import NasaDataset
from utilities import *
from cam import CAM

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model_name', type=str,  default='cloudunet', help='Model Name')
    parser.add_argument('--batch_size', type=int,  default=1, help='Batch Size')
    parser.add_argument('--lr', type=float,  default=0.01, help='the learning rate')
    parser.add_argument('--scheduler', type=str,  default=None, help='Scheduler')
    parser.add_argument('--func', type=str,  default=None, help='loss function')
    parser.add_argument('--w1', type=float,  default=1.0, help='weighting factor')
    parser.add_argument('--w2', type=float,  default=1.0, help='weighting factor')
    parser.add_argument('--w3', type=float,  default=1.0, help='weighting factor')
    parser.add_argument('--pretrained', type=str,  default=None, help='Pretrained Model Name')
    parser.add_argument('--patch_size', type=int,  default=20, help='Patch Size')
    parser.add_argument('--p_batch_size', type=int,  default=128, help='Patch Batch Size')
    parser.add_argument('--stride', type=int,  default=10, help='Patch Size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--num_of_gpu', type=int, default=1, help='Number of GPUs to use')
    args = parser.parse_args()
    return args

def main():
    model_filenames=[]
    # Parse the arguments
    args = parse_args()

    # Set CUDA_VISIBLE_DEVICES to limit the GPUs used
    if args.num_of_gpu > 0:
           os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.num_of_gpu))

    # Check available GPUs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_available_gpus = torch.cuda.device_count()
    print(f"Running on {num_available_gpus} GPU(s): {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")


#    model_filenames=[]
#    # Parse the arguments
#    args = parse_args()

#    # Check if the GPU is available
#    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
#    print(f'Main Selected device: {device}')


    lr   = args.lr
    # Load and Batch the Data
    batch_size = args.batch_size
    model_name = args.model_name
    patch_size = args.patch_size

    cv_train_loss = []
    cv_valid_loss = []
    cv_test_cer_loss = []
    cv_test_cot_loss = []

    # dataset dir
    dataset_dir1 = "/home/ec2-user/Refl"
    dataset_dir2 = "/home/ec2-user/ncer_fill3"
    #cv_train_list = np.load("Data_split/train_split_100m.npy")
    #cv_valid_list = np.load("Data_split/valid_split_100m.npy")
    #cv_test_list = np.load("Data_split/test_split_100m.npy")

    # modify above 3 lines for aws test
    cv_train_list = np.arange(1,31)
    cv_valid_list = np.arange(31,41)
    cv_test_list = np.arange(41,51)

    for fold in range(1): ## changed to  5  to be faster
        cp=False

       # if model_name =="cam":
       #     model = NetCBAM4(in_channels=2,gate_channels=64,n_classes=2)
       # elif model_name=="cloudunet":
       #     model = CloudUNet(n_channels=2,n_classes=2)
        if model_name=="cam":
           model =CAM(in_channels=2,gate_channels=64)
       # elif model_name=="cam9p":
       #     model =CAM9p(in_channels=2,gate_channels=64)
       # elif model_name=="cam11":
        #    model =CAM11(in_channels=2,gate_channels=64)        
        #elif model_name=="cam12":
        #    model =CAM12(in_channels=2,gate_channels=64)         
        #elif model_name=="nataraja":
       #     model = Nataraja(n_channels=2,n_classes=2)
       # elif model_name=="maresunet":
        #    model = MAResUNet(2,2)
       # elif model_name == "okamura":
       #     model = DNN2w2(n_channels=2)
       #     cp=1
       # elif model_name == "okamura2":
       #     model = DNN2w64(n_channels=2)
       #     cp=2
       # elif model_name=="cam9s":
       #     model =CAM9s(in_channels=2,gate_channels=64)
        model = initialize_weights(model)
        # Train model with five fold cross validation
        train_data = NasaDataset(fold=fold,profilelist= cv_train_list,
                                root_dir1=dataset_dir1,root_dir2=dataset_dir2,
                                patch_size=patch_size,stride=args.stride, cp=cp)
        valid_data = NasaDataset(fold=fold,profilelist= cv_valid_list,
                                root_dir1=dataset_dir1,root_dir2=dataset_dir2,
                                patch_size=patch_size,stride=args.stride, cp=cp)
        test_data = NasaDataset(fold=fold,profilelist= cv_test_list,
                                root_dir1=dataset_dir1,root_dir2=dataset_dir2,
                                patch_size=patch_size,stride=args.stride, cp=cp)

        train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size,shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=False)

        # Use DataParallel to utilize multiple GPUs
        model = nn.DataParallel(model)

        # Move the model to GPU
        model = model.to(device)
        
        # Create a directory to save the model and other relevant information
        saved_model_dir =os.path.join('v71_saved_model',model_name)
        try:
            os.makedirs(saved_model_dir)
        except FileExistsError:
            print("folder already exists")

            
        # Generate log file, and log environment information
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(saved_model_dir, f'train_{timestamp}.log')
        log_level = 1
        logger = get_root_logger(log_file=log_file,log_level=log_level)

        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        model_summary = summary(model, input_size=(128,2, args.patch_size, args.patch_size))
        
        logger.info((model_summary))  

        # Generate saved model name
        saved_model_name = model_name+"_fold_"+str(fold)+'_'+timestamp+'.pth'
        saved_model_path = os.path.join(saved_model_dir,saved_model_name)
        model_filenames.append(saved_model_name)
        # Define the parameters
        params = {
            "saved_model_path" : saved_model_path,
            "batch"            : batch_size,
            "optimizer"        : "SGD",
            "lr"               : lr,
            "loss"             : args.func,
            "scheduler"        : args.scheduler,
            "num_epochs"       : 2,
            "patience"         : 1,
            "w1"               : args.w1,
            "w2"               : args.w2,

            "p_batch_size"     : args.p_batch_size,
            "patch_size"       : args.patch_size
        }


        msg = "Predicts cot,CER from r1 and r2 "

        logger.info(msg)

        # Log these parameters in the log file
        param_info ='\n'.join([(f'{k}: {v}') for k, v in params.items()])
        logger.info('Parameters info:\n' + dash_line + param_info + '\n' +
                    dash_line)



        ########################################## Start Training ############################################
        model, train_loss, valid_loss = train_model(model,train_loader,valid_loader, params, device,log_level)
        train_loss2 = valid_loss[len(train_loss)-params['patience']-1]
        valid_loss2 = valid_loss[len(valid_loss)-params['patience']-1]


        # Visualizing the Loss and the Early Stopping Checkpoint
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
        plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

        # find position of lowest validation loss
        minposs = valid_loss.index(min(valid_loss))+1 
        plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, 4) # consistent scale
        plt.xlim(0, len(train_loss)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        figname = os.path.join(saved_model_dir,'loss_plot_'+ model_name+"_fold_"+'_'+timestamp+'.png')
        fig.savefig(figname, bbox_inches='tight')
        # # Test the Trained Model
        # test_loss = test_model(model, test_loader,params,device,log_level)
        test_cot, test_cer = test_model2(model, test_loader,device,args.patch_size,log_level)


        cv_train_loss.append(train_loss2)
        cv_valid_loss.append(valid_loss2)
        cv_test_cot_loss.append(test_cot)
        cv_test_cer_loss.append(test_cer)

    stop = timeit.default_timer()

    print_msg = (f'Time: {(stop-start):.2f}')
    logger.info(print_msg)
    a1 = np.average(cv_train_loss)
    a2 = np.average(cv_valid_loss)
    a3 = np.average(cv_test_cot_loss)
    a4 = np.average(cv_test_cer_loss)

    print_msg = (
    f'  Train Loss:  {a1:.4f}\n' +
    f'  Valid loss:  {a2:.3f}\n' + 
    f'  Test COT MSE: {a3:.4f}\n '+
    f'  Test CER MSE: {a4:.4f} ')
    logger.info(print_msg)
    return {"model_name": model_name,"saved_model_dir":saved_model_dir,"logfile":timestamp,
            "model_filenames":model_filenames}




if __name__=="__main__":
    out = main()
    # run_test(out,stride=4)
