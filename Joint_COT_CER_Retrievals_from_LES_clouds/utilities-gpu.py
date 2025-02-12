'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
    
    GPU enabled by: Seraj Mostafa
    PhD Candidate, BDALab
'''

'''
Holds the latest utilities functions. 
Includes early stopping and scheduling operation as well.
'''

# import EarlyStopping
from random import random
from pytorchtools import EarlyStopping

import torch
import torch.nn as nn
from torch.optim import Adam,SGD, lr_scheduler
import numpy as np

from mmcv.utils import collect_env as collect_base_env
import logging
from mmcv.utils import get_logger, print_log


from losses import *
from metrics import *

def initialize_weights(model):
    # Iterate over the model's parameters and initialize them
    for param in model.parameters():
        nn.init.normal_(param, mean=0, std=1)
    return model

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





def train_model(model, train_loader, valid_loader, params, device,log_level):

    # mse_loss_fn = torch.nn.MSELoss()
    # Assign parameters to variables
    n_epochs        = params['num_epochs']
    lr              = params['lr']
    saved_model_path = params["saved_model_path"] 
    patience        = params["patience"] 
    p_batch_size    = params["p_batch_size"]
     
    ### Define the loss function
    if params['loss']=="MSE":
        criterion = torch.nn.MSELoss()
    elif params['loss']=="BCE":
        criterion = torch.nn.BCELoss()
    elif params['loss']=="dice":
        criterion = soft_DiceLoss()
    elif params['loss']=="iou":
        criterion = soft_IoULoss()
    elif params['loss']=="bce_iou":
        criterion = BCE_IoULoss(params["w1"] )
        
    elif params['loss']=="Joint_L2":
        w1           = params["w1"]
        w2           = params["w2"]
        criterion = Joint_L2(w1,w2)
    elif params['loss']=="Joint_L2_Binary":
        w1           = params["w1"]
        w2           = params["w2"]
        w3           = params["w3"]
        criterion    = Joint_L2_Binary(w1,w2,w3)
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

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 



    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True,path=saved_model_path)
    
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        # model.to(device)
        model.train() # prep model for training
        for _, data in enumerate(train_loader, 1):
        # for i in range(len(train_loader.dataset)):
        #     data = train_loader.dataset[i]
            r_train, m_train = data['rad_patches'],data['cot_patches']

            # Iterate through the list in batches
            # for p_b in range(0, len(r_train), p_batch_size):
            for p_b in range(0, len(r_train)):
                # tensor_list1 = r_train[p_b:p_b+p_batch_size]
                # X_train      = torch.stack(tensor_list1, dim=0)
                X_train        = r_train[p_b]
                # tensor_list2 = m_train[p_b:p_b+p_batch_size]
                # Y_train      = torch.stack(tensor_list2, dim=0)
                Y_train        = m_train[p_b]
                
                # Process the current batch
                # print(f"Processing batch {p_b//p_batch_size + 1}: {p_batch_size}")
            
                # Y_train = torch.unsqueeze(Y_train,1)
                # Move tensor to the proper device
                X_train = X_train.to(device,dtype=torch.float)
                Y_train = Y_train.to(device,dtype=torch.float)

                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(X_train)
                # print(Y_train.shape)
                # print(output.shape)
                # calculate the loss
                loss = criterion(output, Y_train)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # record training loss
                train_losses.append(loss.item())
                # print(train_losses[p_b])
                

           
        lr_info =optimizer.param_groups[0]['lr']
        ######################    
        # validate the model #
        ######################
        valid_loss = test_model(model,valid_loader,params,device,log_level)

        if scheduler:
            scheduler.step(valid_loss) 
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'lr: {lr_info}  '                )
        
        logger.info(print_msg)
        if epoch%10==0:
            print(print_msg)

        
        # clear lists to track next epoch
        train_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(saved_model_path))

    return  model, avg_train_losses, avg_valid_losses



def test_model(model, test_loader,params,device,log_level=None):
    '''
    measures each image separately
    '''
    p_batch_size    = params["p_batch_size"]
    # initialize lists to monitor test loss and accuracy
    test_losses = []

    # model.to(device)
    model.eval() # prep model for evaluation
    ### Define the loss function
    if params['loss']=="MSE":
        criterion = torch.nn.MSELoss()
    elif params['loss']=="BCE":
        criterion = torch.nn.BCELoss()
    elif params['loss']=="dice":
        criterion = soft_DiceLoss()
    elif params['loss']=="iou":
        criterion = soft_IoULoss()
    elif params['loss']=="bce_iou":
        criterion = BCE_IoULoss(params["w1"] )
    elif params['loss']=="Joint_L2":
        w1           = params["w1"]
        w2           = params["w2"]
        criterion = Joint_L2(w1,w2)
    elif params['loss']=="Joint_L2_Binary":
        w1           = params["w1"]
        w2           = params["w2"]
        w3           = params["w3"]
        criterion    = Joint_L2_Binary(w1,w2,w3)
    # X_train = torch.rand((1,2,10,10),dtype=torch.float)
    # Y_train = torch.rand((1,1,10,10),dtype=torch.float)
    # for i in range(len(test_loader.dataset)):
    for i in range(len(test_loader.dataset)):
    # for _, data in enumerate(test_loader, 1):
        data = test_loader.dataset[i]
        # get the data
        r_test, m_test = data['rad_patches'],data['cot_patches']


        # Iterate through the list in batches
        for p_b in range(0, len(r_test), p_batch_size):
        # for p_b in range(0,len(r_test)):
            tensor_list1= r_test[p_b:p_b+p_batch_size]
            X_test = torch.stack(tensor_list1, dim=0)
            tensor_list2 = m_test[p_b:p_b+p_batch_size]
            Y_test = torch.stack(tensor_list2, dim=0)

            # X_test = r_test[p_b]
            # Y_test = m_test[p_b]


            # X_test = torch.unsqueeze(X_test,0)
            # Y_test = torch.unsqueeze(Y_test,0)
            # Y_test = torch.unsqueeze(Y_test,0)

            # Move tensor to the proper device
            X_test = X_test.to(device,dtype=torch.float)
            Y_test = Y_test.to(device,dtype=torch.float)
            # X_test = X_test.to(dtype=torch.float)
            # Y_test = Y_test.to(dtype=torch.float)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(X_test)
            # print(Y_test.shape)
            # print(output.shape)
            # calculate the loss
            loss = criterion(output, Y_test)

            # update test loss 
            test_losses.append(loss.item())


    test_loss = np.average(test_losses)


    # print_msg = ('Test Loss: {:.6f}\n'.format(test_loss))
    # print(print_msg)
    # if log_level:
    #     logger = get_root_logger(log_level=log_level)
    #     logger.info(print_msg)
    return test_loss

def test_model2(model, test_loader,device,p_batch_size = 20,log_level=None):
    '''
    measures each image separately
    '''
    
    # initialize lists to monitor test loss and accuracy
    test_cot = []
    test_cer = []

    criterion =torch.nn.MSELoss()
    # model.to(device)
    model.eval() # prep model for evaluation
    ### Define the loss function
    # X_train = torch.rand((1,2,10,10),dtype=torch.float)
    # Y_train = torch.rand((1,1,10,10),dtype=torch.float)
    for i in range(len(test_loader.dataset)):
        data = test_loader.dataset[i]
        # get the data
        r_test, m_test = data['rad_patches'],data['cot_patches']

        # Iterate through the list in batches
        for p_b in range(0, len(r_test), p_batch_size):
        # for p_b in range(0, len(r_train)):
            tensor_list1= r_test[p_b:p_b+p_batch_size]
            X_test = torch.stack(tensor_list1, dim=0)
            tensor_list2 = m_test[p_b:p_b+p_batch_size]
            Y_test = torch.stack(tensor_list2, dim=0)
            # X_test = r_test[p_b]
            # Y_test = m_test[p_b]

            # X_test = torch.unsqueeze(X_test,0)
            # Y_test = torch.unsqueeze(Y_test,0)
            # Y_test = torch.unsqueeze(Y_test,0)

            # Move tensor to the proper device
            X_test = X_test.to(device,dtype=torch.float)
            Y_test = Y_test.to(device,dtype=torch.float)

            # X_test = X_test.to(dtype=torch.float)
            # Y_test = Y_test.to(dtype=torch.float)
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(X_test)
            # print(output.shape)
            # print(Y_test.shape)

            loss_COT = criterion(output[:,0,:,:], Y_test[:,0,:,:])
            loss_CER = criterion(output[:,1,:,:], Y_test[:,1,:,:])

            # # print(output.shape)
            # # calculate the acc
            # y_pred_binary = (output[:,2,:,:] > 0.5).float() 
            # pixel_acc = pixel_accuracy(Y_test[:,2,:,:],y_pred_binary)
            # mean_iou  = compute_IoU(Y_test[:,2,:,:],y_pred_binary)

            # # update test acc 
            # test_acc.append(pixel_acc)
            # test_iou.append(mean_iou)

            # update the losses
            test_cot.append(loss_COT.item())
            test_cer.append(loss_CER.item())

    # avg_test_acc = np.average(test_acc)
    # avg_test_iou = np.average(test_iou)

    avg_test_cot = np.average(test_cot)
    avg_test_cer = np.average(test_cer)


    # print_msg1 = ('Test Pixel Acc: {:.6f}\n'.format(avg_test_acc))
    # print(print_msg1)

    # print_msg2 = ('Test IoU: {:.6f}\n'.format(avg_test_iou))
    # print(print_msg2)

    # if log_level:
    #     logger = get_root_logger(log_level=log_level)
    #     logger.info(print_msg1)
    #     logger.info(print_msg2)

    return avg_test_cot,avg_test_cer







def get_predictions(model,X_test,Y_test=None,device='cpu'):
    '''
    measures each image separately
    X_test=numpy array , dim (10,10,2)
    Y_test=numpy array,  dim(10,10,2) or (6,6,2)
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval() # prep model for evaluation

    # Move tensor to the proper device
    X_test = torch.unsqueeze(X_test,0)
    # Y_test = torch.unsqueeze(Y_test,0)

    X_test = X_test.to(device,dtype=torch.float)
    # Y_test = Y_test.to(device,dtype=torch.float)

    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(X_test)
     
    predictions =output.data.cpu().numpy()
    y_pred = np.squeeze(predictions)
    
    return y_pred
    # return (y_pred > 0.5)*1



    
if __name__=="__main__":
    # np.random.seed(5)
    r = np.random.randint(1,5,size = (100,10,10,2))
    c = np.random.randint(1,5,size = (100,10,10,1))


    # # X_train,X_valid,X_test,Y_train,Y_valid,Y_test = cross_val(r,c)

    # a = np.random.randint(1,10,size = (2,2))
    # b = np.random.randint(1,10,size = (2,2))
    # mse = (np.square(a - b)).mean()
    # print(mse)
    # print("Done!")
