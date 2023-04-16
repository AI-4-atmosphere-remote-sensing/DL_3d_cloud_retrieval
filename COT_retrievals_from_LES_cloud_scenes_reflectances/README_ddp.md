## PyTorch Distributed Data Parallel Implementation for Cloud Optical Thickness Retrieval Application

# Step 1: Taki configuration:
1. The user must be having a taki account
2. The user must be able to access the account with their credentials.
3. Open the Windows PowerShell, and access your account as given example below: <br />
    ssh garimak1@taki.rs.umbc.edu <br />
    Enter your password <br />
4. You should be able to log in to the taki cluster successfully.

# Step 2: Conda Environment Set Up:
1. Set up a directory to install conda. <br />
2. Use this URL to install the conda: https://docs.conda.io/en/latest/miniconda.html <br />
    2.1 Detailed steps to install conda set up is mentioned below: <br />
    2.2 Navigate to URL: https://docs.conda.io/en/latest/miniconda.html <br />
    2.3 Download and save the Python 3.9 window installer. <br />
    2.4 Go to the terminal and the directory where to want to install the conda. <br />
    2.5 Upload the downloaded file to the directory. <br />
    2.6 To install it with our platform we have a command : <br />
        sh <filename> <br />
        for eg: sh  Miniconda3-latest-Linux-x86_64.sh <br />
3. All the packages should start installing 
4. Press yes and press enter and the system will try to unpack the miniconda installer for Linux.
5. Now you can check in the installed packages Python must be downloaded and also some other associated packages.
6. Press yes wherever it asks you to proceed further while installing.
7. If your setup is done properly, you should receive a message saying: "Thank you for installing Miniconda3!".
8. Now try to create a conda environment using the command: conda create –name cot python=3.9   <br />
Note: Here cot is a random conda environment you can specify any name as per your requirement.  <br />
      This command will help to set up your python virtual environment for 3.9
9. After completing this step we need to activate the newly created conda environment, we can do this by this command:<br />
    conda activate cot

# Step 3: Install the required Packages
 Once the conda environment set is done completely, the next step is to install all the dependencies under the newly created conda environment. <br />
 Below is the list of dependencies we need to install before jumping to the execution of the model training.
1. Install Pytorch: conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
2. Install Torchinfo (1.7.1): pip install torchinfo
3. Install matplotlib (3.6.1): pip install matplotlib
4. Install mmcv (1.6.2): pip install mmcv
5. Install scikit-learn (): pip install scikit-learn
6. Install h5py: pip install h5py

# Step 4: Clone the Source Code
1. Create a directory 
2. Navigate to the project directory and clone the COT_Retrieval project: https://github.com/AI-4-atmosphere-remote-sensing/DL_3d_cloud_retrieval.git
Note: The major changes made in the main_ddp.py and utilities_ddp.py.

# Step 5: Data Preparation
The following bash script processes the raw data which is in hdf format into numpy arrays. It also creates a five fold cross validated dataset. The scripts requires the dataset path. <br />
For data preprocessing, run the following command. <br />
bash data_preprocess.sh $data_path <br />
Note: The data is present in the “data” folder, we need to replace the “$data_path”  with the path of the data folder.
The data_path will be different for every user for eg for me the path is : /umbc/rs/nasa-access/users/garimak1/ddp/COT_retrievals_from_LES_cloud_scenes_reflectances/data
 bash data_preprocess.sh /umbc/rs/nasa-access/users/garimak1/ddp/COT_retrievals_from_LES_cloud_scenes_reflectances/data/

# Step 6: To execute the slurm file use the following command: sbatch train.slurm
Creating a slurm file: A Slurm file is a script used to organize and run distributed training jobs over numerous nodes in a high-performance computing (HPC) cluster when using PyTorch DDP (Distributed Data Parallel) training. <br />
Below is the slurm file used for COT retrieval project: <br />
#!/bin/bash <br />
#SBATCH --job-name=128_2cot  <br />
#SBATCH --output=128_2slurm.out <br />
#SBATCH --error=128_2slurm.err <br />
#SBATCH --partition=gpu2018 <br />
#SBATCH --qos=high_mem <br />
#SBATCH --time=25:30:00  <br />
#SBATCH --gres=gpu:2 <br />
#SBATCH --nodes=1 <br />
#SBATCH --tasks-per-node=2 <br />
nvidia-smi <br />
export CUDA_LAUNCH_BLOCKING=1 <br />
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --model_name okamura --batch_size 128 --lr 0.01 --world_size 2 <br />

The distributed training task utilizing PyTorch DDP is really conducted with this command. It employs torchrun, a tool for starting distributed PyTorch operations. The —standalone and —nnodes=1 and —nproc per node=2 parameters define the number of nodes and processes per node, The Python script that implements the training code is called main.py. The remaining flags (—model name, —batch size, —lr, and —world size) define the training job’s hyperparameters.

# Step 7: Command to execute the application.
1. Use this command : sbatch <slurm file>
    for eg: sbatch <train_ddp.slurm>
2. A job should be created after running this command.

# Step 7: Observe the results:
  1. Check the train_ddp.slurm file. <br />
  2. Check the job name, output and error file names.  <br />
  3. Slurm.out file must be created, once the job is successfully implemented. for eg: 128_2slurm.out<br />
  4. Slurm.err file must be created, once the job is successfully implemented. for eg: 128_2slurm.err<br />

# Step 8: For the output open the slurm.out file use the following command:
 more slurm.out
