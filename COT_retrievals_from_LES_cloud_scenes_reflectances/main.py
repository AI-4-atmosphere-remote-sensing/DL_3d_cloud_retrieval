# Import libraries
import timeit

start = timeit.default_timer()

# import libraries
import argparse
import os
from torchinfo import summary
import time
import torch
from torch.utils.data import  DataLoader
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)

from utilities import NasaDataset
from utilities import train_model,test_model, get_root_logger,collect_env
from model_config import DNN2w, EncoderDecoder, DnCNN, DnCNNmod, CloudUNet 


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--device', type=int,  default=None, help='CUDA')
    parser.add_argument('--model_name', type=str,  default=None, help='Model Name')
    parser.add_argument('--batch_size', type=int,  default=None, help='Batch Size')
    parser.add_argument('--lr', type=float,  default=None, help='the learning rate')
    args = parser.parse_args()
    return args

def main():
    # Parse the arguments
    args = parse_args()

    # Check if the GPU is available
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    print(f'Main Selected device: {device}')


    lr   = args.lr
    # Load and Batch the Data
    batch_size = args.batch_size
    model_name = args.model_name

    # Train model with five fold cross validation
    data_dir="preprocessed_data/Cloud_25"
    dataset_name = "cloud_25"

    # To store the five fold losses
    cv_valid_losses = []
    cv_valid_mse_losses = []
    cv_test_losses   = []
    for i in range (5):

        # Specify the Model
        cp = False

        if model_name=="okamura":
            model = DNN2w()
            cp = True
        elif model_name=="encoderdecoder":
            model = EncoderDecoder(64)
        elif model_name=="dncnn":
            model = DnCNN(2,5)
        elif model_name=="dncnn_mod":
            model = DnCNNmod(2,5)
        elif model_name =="cloudunet":
            model = CloudUNet(n_channels=2,n_classes=1)

        # Create a directory to save the model and other relevant information
        saved_model_dir =os.path.join('saved_models',model_name)
        try:
            os.makedirs(saved_model_dir)
        except FileExistsError:
            print("folder already exists")

        fname_X_train = data_dir+"/X_train_patch_%01d.npy"%(i)
        fname_X_valid = data_dir+"/X_valid_patch_%01d.npy"%(i) 
        fname_X_test  = data_dir+"/X_test_patch_%01d.npy"%(i)
        fname_Y_train = data_dir+"/Y_train_patch_%01d.npy"%(i)
        fname_Y_valid = data_dir+"/Y_valid_patch_%01d.npy"%(i)
        fname_Y_test  = data_dir+"/Y_test_patch_%01d.npy"%(i)

        X_train = np.load(fname_X_train)
        X_valid = np.load(fname_X_valid)
        X_test = np.load(fname_X_test)
        Y_train = np.load(fname_Y_train)
        Y_valid = np.load(fname_Y_valid)
        Y_test = np.load(fname_Y_test)

        train_data = NasaDataset(dataset_name=dataset_name,fold =i, X=X_train, Y= Y_train,cropped=cp,transform=True)
        valid_data = NasaDataset(dataset_name=dataset_name,fold =i, X=X_valid, Y= Y_valid,cropped=cp,transform=True)
        test_data  = NasaDataset(dataset_name=dataset_name,fold =i, X=X_test, Y= Y_test,cropped=cp,transform=True)



        train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size,shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=True)




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
        model_summary = summary(model, input_size=(batch_size,2, 10, 10))
        logger.info((model_summary))  

        # Generate saved model name
        saved_model_name = model_name+"_fold_"+str(i)+'_'+timestamp+'.pth'
        saved_model_path = os.path.join(saved_model_dir,saved_model_name)
        # Define the parameters
        params = {
            "saved_model_path":saved_model_path,
            "batch_size":batch_size,
            "optimizer":"Adam",
            "lr":lr,
            "loss": "MSE",
            "scheduler" :"ReduceLR",
            "num_epochs" : 5,
            "patience": 2,
            "dataset_name": dataset_name # early stopping patience;
        }


        # Log these parameters in the log file
        param_info ='\n'.join([(f'{k}: {v}') for k, v in params.items()])
        logger.info('Parameters info:\n' + dash_line + param_info + '\n' +
                    dash_line)



        ########################################## Start Training ############################################
        model, train_loss, valid_loss, valid_mse_loss = train_model(model,train_loader,valid_loader, params, device,log_level)
        cv_valid_losses.append(valid_loss[len(valid_loss)-params['patience']-1])
        cv_valid_mse_losses.append(valid_mse_loss[len(valid_mse_loss)-params['patience']-1])


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
        plt.ylim(0, 100) # consistent scale
        plt.xlim(0, len(train_loss)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        figname = os.path.join(saved_model_dir,'loss_plot_'+ model_name+"_fold_"+str(i)+'_'+timestamp+'.png')
        fig.savefig(figname, bbox_inches='tight')
        # Test the Trained Model

        test_loss,_,_,_ = test_model(model, test_loader,params,device,log_level)
        cv_test_losses.append(test_loss)
        stop = timeit.default_timer()

        print_msg = (f'Time: {(stop-start):.2f}')
        logger.info(print_msg)
    
    mean  = np.mean(cv_valid_losses)
    std   = np.std(cv_valid_losses)

    mean_test_loss = np.mean(cv_test_losses)
    std_test_loss  = np.std(cv_test_losses)

    mse_mean = np.mean(cv_valid_mse_losses)
    mse_std  = np.std(cv_valid_mse_losses)

    print_msg = (f'Mean Valid Loss: {mean:.3f}'+f'  Std Valid Loss: {std:.3f}'+
    f'  Mean Test loss:  {mean_test_loss:.3f}' +f'  Std Test loss: {std_test_loss:.3f} '+
    f'   Mean Valid MSE Loss: {mse_mean:.3f}'  + f'   Std Valid MSE Loss: {mse_std:.3f}')
    logger.info(print_msg)




if __name__=="__main__":
    main()
