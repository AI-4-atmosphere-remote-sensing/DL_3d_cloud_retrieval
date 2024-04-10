# Import libraries
import argparse
import numpy as np
from .dataloader import NasaDataset
from pytorchtools import EarlyStopping

import torch
from torch.optim import Adam,SGD, lr_scheduler
from torch.utils.data import DataLoader

import wandb

import logging
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_logger

from torch.nn.parallel import DistributedDataParallel as DDP

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# start process 0

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"




def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data_dir', type=str, default="/home/local/AD/ztushar1/Data/cloud_fraction_100LES_sza60/", help='dataset directory')
    args = parser.parse_args()
    return args

def compute_cloud_fraction(image,limit):
    mask = (image>limit)*1
    fraction = np.mean(mask)
    return fraction

def get_mean_and_std_input(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for batch, sample in enumerate (loader):
        data = sample['reflectance']
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

def get_mean_and_std_output(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for batch, sample in enumerate (loader):
        data = sample['cot']
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return (self.mse(torch.log(pred + 1-torch.min(pred)), torch.log(actual + 1-torch.min(actual))))

def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    return env_info

def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
    Returns:
        :obj:`logging.Logger`: The obtained logger.
    """
    return get_logger('cloud_retrieval', log_file, log_level)


def train_model(model, train_loader, valid_loader, params,device,log_level):

    # Assign parameters to variables
    n_epochs        = params['num_epochs']
    lr              = params['lr']
    saved_model_path = params["saved_model_path"] 
    patience        = params["patience"] 
    
    #device_id = 0  # set the device ID to 0

    #device = torch.device("cuda", device_id)
    #torch.cuda.set_device(device)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("torch.cuda.is_available:", torch.cuda.is_available())
    model = model.to(device)
    #model = torch.nn.parallel.DistributedDataParallel(model, device=[device])
    model = DDP(model,device_ids=[device], find_unused_parameters=True)
        
    ### Define the loss function
    if params['loss']=="MSE":
        criterion = torch.nn.MSELoss()
    elif params['loss'] == "L1Loss":
        criterion = torch.nn.L1Loss()
    elif params['loss']=="MSLE":
        criterion = MSLELoss()

    # specify optimizer
    if params['optimizer']=="Adam":
        optimizer = Adam(model.parameters(), lr=lr,weight_decay=1e-05)
    elif params['optimizer']=="SGD":
        optimizer = SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=1e-05)

    if params['scheduler']=="ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9,verbose=True)
    elif params['scheduler']=="ReduceLR":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,threshold=1e-3,verbose=True)
    elif params['scheduler']=="StepLR":
        scheduler = lr_scheduler.StepLR(optimizer=optimizer,step_size=20,gamma=0.95)
    else:
        scheduler = None   

    logger = get_root_logger(log_level=log_level)
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 

    # to track the validation mse loss as the model trains for model comparison
    # valid_mse_losses = []
    avg_valid_mse_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True,path=saved_model_path)
    import time
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 0
    for epoch in range(1, n_epochs + 1):
        gpu_start_times = [time.time() for _ in range(num_gpus)]

        ###################
        # train the model #
        ###################
        print(f'Number of epochs: {n_epochs}')
        model.to(device)
        model.train() # prep model for training
        for batch, data in enumerate(train_loader, 1):
            # get the data
            X_train, Y_train = data['reflectance'][:,0:2,:,:] ,data['cot'][:,0,:,:] 
            Y_train = torch.unsqueeze(Y_train,1)
            # Move tensor to the proper device
            X_train = X_train.to(device,dtype=torch.float)
            Y_train = Y_train.to(device,dtype=torch.float)
            # print(Y_train.shape)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(X_train)
            # calculate the loss
            loss = criterion(output, Y_train)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

            wandb.log({"epoch": epoch, "train_loss": loss.item()})


           
        lr_info =optimizer.param_groups[0]['lr']
        ######################    
        # validate the model #
        ######################
        valid_loss,_,_,valid_mse_loss = test_model(model,valid_loader,params,log_level)


        if scheduler:
            scheduler.step(valid_loss) 

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        # clear lists to track next epoch
        train_losses = []
        # store the avg train and validation loss per epoch.
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # For model comparison with different loss criterion
        avg_valid_mse_losses.append(valid_mse_loss)
        gpu_end_times = [time.time() for _ in range(num_gpus)]

        

        # Print Training statistics  
        epoch_len = len(str(n_epochs))      
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'lr: {lr_info}  '                 +
                     f'valid_mse_loss: {valid_mse_loss:.5f}')
        
        for i in range(num_gpus):
            gpu_start_time = gpu_start_times[i]
            gpu_end_time = gpu_end_times[i]
            gpu_time = gpu_end_time - gpu_start_time
            print_msg += f' gpu{i}_time: {gpu_time:.2f}s (start: {gpu_start_time:.2f}, end: {gpu_end_time:.2f})'

        
        logger.info(print_msg)
        if epoch%10==0:
            print(print_msg)        

        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(saved_model_path))

    return  model, avg_train_losses, avg_valid_losses, avg_valid_mse_losses


def test_model(model, test_loader,params,device,log_level=None):
    '''
    measures each image separately
    '''

    # initialize lists to monitor test loss and accuracy
    mse_loss_fn = torch.nn.MSELoss()
    test_losses = []
    mse_losses  = []
    predictions = []
    device_id = 0  # set the device ID to 0

    #device = torch.device("cuda", device_id)
    #torch.cuda.set_device(device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("torch.cuda.is_available:", torch.cuda.is_available())
    model.to(device)
    model.eval() # prep model for evaluation
    ### Define the loss function
    if params['loss']=="MSE":
        criterion = torch.nn.MSELoss()
    elif params['loss'] == "L1Loss":
        criterion = torch.nn.L1Loss()
    elif params['loss']=="MSLE":
        criterion = MSLELoss()
    # X_train = torch.rand((1,2,10,10),dtype=torch.float)
    # Y_train = torch.rand((1,1,10,10),dtype=torch.float)
    for i in range(len(test_loader.dataset)):
        data = test_loader.dataset[i]
        # get the data
        X_test, Y_test = data['reflectance'][0:2,:,:] ,data['cot'][0,:,:] 
        # Move tensor to the proper device
        X_test = torch.unsqueeze(X_test,0)
        Y_test = torch.unsqueeze(Y_test,0)
        Y_test = torch.unsqueeze(Y_test,0)
        X_test = X_test.to(device,dtype=torch.float)
        Y_test = Y_test.to(device,dtype=torch.float)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(X_test)
        # calculate the loss
        loss = criterion(output, Y_test)
        loss_mse = mse_loss_fn(output, Y_test)
        # update test loss 
        test_losses.append(loss.item())
        predictions.append(output.data)

        mse_losses.append(loss_mse.item())

    test_loss = np.average(test_losses)
    mse_loss  = np.average(mse_losses)

    wandb.log({"test_loss": test_loss, "mse_loss": mse_loss})

    print_msg = ('Test Loss: {:.6f}\n'.format(test_loss))
    print(print_msg)
    if log_level:
        logger = get_root_logger(log_level=log_level)
        logger.info(print_msg)
    return test_loss,test_losses, predictions,mse_loss


def get_pred(model,X_test,Y_test,device='cuda'):
    '''
    measures each image separately
    X_test=numpy array , dim (10,10,2)
    Y_test=numpy array,  dim(10,10,1) or (6,6,1)
    '''
    criterion = torch.nn.MSELoss()
    #device_id = 0  # set the device ID to 0

    #device = torch.device("cuda", device_id)
    #torch.cuda.set_device(device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("torch.cuda.is_available:", torch.cuda.is_available())
    model.to(device)
    model.eval() # prep model for evaluation

    # Move tensor to the proper device
    X_test = torch.unsqueeze(X_test,0)
    Y_test = torch.unsqueeze(Y_test,0)
    Y_test = torch.unsqueeze(Y_test,0)
    X_test = X_test.to(device,dtype=torch.float)
    Y_test = Y_test.to(device,dtype=torch.float)

    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(X_test)

    # Only for thsholding of log of cot model, not for log of cot+thresholding
    # output[output>5.8882]=5.8882
    # output[output<-4.6052]=-4.6052
    # calculate the loss
    loss = criterion(output, Y_test)
    predictions =output.data.cpu().numpy()
    y_pred = np.squeeze(predictions)

    test_loss = loss.item()

    print_msg = ('Test Loss: {:.3f}\n'.format(test_loss))
    # print(print_msg)

    return test_loss,y_pred



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


        train_data = NasaDataset(dataset_name ='cloud_25', fold = i, X=X_train, Y= Y_train,cropped=False, transform=False)
        loader = DataLoader(train_data, batch_size=10)

        mean, std = get_mean_and_std_input(loader)
        mean, std = get_mean_and_std_output(loader)


        print("Dataset Mean: ",mean)
        print("Dataset Std: ",std)
