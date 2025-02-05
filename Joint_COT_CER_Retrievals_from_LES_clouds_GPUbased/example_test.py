import yaml
import argparse
import h5py
import numpy as np
from cam import CAM
from run_test import *
from visualization import *

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_inference(config):
    #1. load model
    model_name = config['model']['name']
    if model_name=="cam":
        model =CAM(in_channels=2,gate_channels=64)
    
    # Conditionally load checkpoint
    if config['model']['pretrained'] and 'checkpoint' in config['test']:
        checkpoint_path = config['test']['checkpoint']
        print(f"Loading checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path,map_location=torch.device('cpu')))
    else:
        print("Skipping checkpoint loading.")



    #2. Load test example
    fname = config['data']['path']
    hf = h5py.File(fname, 'r')
    # initialize the array
    X_test = np.empty((144,144,2), dtype=float)
    Y_test = np.empty((144,144,2),dtype=float) 
    temp              = np.nan_to_num(np.array(hf.get("Reflectance_100m_resolution")))
    # reflectance at 0.66 um
    X_test[:,:,0]   = temp[0,:,:]
    # reflectance at 2.13 um
    X_test[:,:,1]   = temp[1,:,:]
    # COT
    Y_test[:,:,0]   = np.nan_to_num(np.array(hf.get("Cloud_optical_thickness_(100m resolution)")))
    # CER
    Y_test[:,:,1] = np.nan_to_num(np.array(hf.get("CER_(100m resolution)")))
    hf.close()



    #3. Get patch based inference
    patch_size = config['data']['patch_size']
    stride = config['data']['stride']
    fold = config['data']['fold']
    Y_test, Y_pred, scores = get_profile_pred(model,model_name,X_test,Y_test,patch_size,stride,fold)
    data = {'profile':Y_test, 'pred':Y_pred}
    return data, scores

def get_visuals(config,data):
    dir_name = config['visuals']['path']
    p_num    = config['visuals']['p_num']

    try:
        os.makedirs(dir_name)
    except FileExistsError:
        print("folder already exists")

    profile = data['profile']
    pred    = data['pred']


    #Scale back CER
    profile[1,:,:] = profile[1,:,:]*30
    pred[1,:,:]    = pred[1,:,:] *30

    use_log=False
    limit1 = [0,7]
    limit2 = [0,40]
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
    

    print("Visuals generated and saved in ",config['visuals']['path'])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    print(config)  # Access values via `config['model']['name']` etc.

    data, scores = get_inference(config)
    print(scores)

    if config['visuals']:
        get_visuals(config,data)
    print("Done!")