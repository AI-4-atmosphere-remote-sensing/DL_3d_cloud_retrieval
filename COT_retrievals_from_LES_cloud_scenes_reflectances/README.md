
# Deep Learning for 3D Cloud Retrievals from LES cloud scences
This GitHub repository contains instructions for preprocessing the raw data, setting up a conda environment, installing necessary packages, and running the train.slurm file to train the model.


Steps to Execute (COT_retrieval)

Step 1: Taki configuration:

1. The user must be having a taki account

2. The user must be able to access the account with their credentials.

3. Open the Windows PowerShell, and access your account as given example below:
ssh garimak1@taki.rs.umbc.edu

4. Enter your password

5. You should be able to log in to the taki cluster successfully.

Step 2: Conda Environment Set Up:

1. Firstly set up a directory to install conda.

2. Use this URL to install the conda: https://docs.conda.io/en/latest/miniconda.html

3. Use the below command to execute the .sh file:- sh Miniconda3-latest-Windows-
x86 64.sh

4. Perform the necessary steps to install all the packages required for building up
the base python environment

5. Check that conda is installed successfully.

6. Now try to create a conda environment using the command:
conda create –name cot python=3.9

Here cot is a random conda environment you can specify any name as per your
requirement.This command will help to set up your python virtual environment
for 3.9

7. After completing this step we need to activate the newly created conda envi-
ronment, we can do this by this command:

conda activate cot

Step 3: Install the required Packages: Once the conda environment set
is done completely, the next step is to install all the dependencies under the newly
created conda environment. Below is the list of dependencies we need to install before
jumping to the execution of the model training.
1. Install Pytorch: conda install pytorch torchvision torchaudio cudatoolkit=11.3
-c pytorch
2. Install Torchinfo (1.7.1): pip install torchinfo
3. Install matplotlib (3.6.1): pip install matplotlib
4. Install mmcv (1.6.2): pip install mmcv
5. Install scikit-learn (): pip install scikit-learn
6. Install h5py: pip install h5py

Step 4: Clone the Source Code: 
Now all the required prerequisite steps
are completed and we will be shifting toward the source code. So, in the selected
directory we need to clone the Project COT retrieval from a specific branch (my-change-LES).

1. Change to the project directory: 
For eg: cd student user/project/ddp/ver0.3/COT retrievals
from LES cloud scenes reflectances/

2. COT retrieval source code from the big-data-lab-umbc repository using git:
https://github.com/AI-4-atmosphere-remote-sensing/DL 3d cloud retrieval.git


Step 5: Data Preparation
The following bash script processes the raw data which is in hdf format into numpy arrays. It also creates a five fold cross validated dataset. The scripts requires the dataset path.

For data preprocessing, run the following command.

bash data_preprocess.sh $data_path



Step 6: To execute the slurm file use the following command:
sbatch train.slurm

Creating a slurm file: A Slurm file is a script used to organize and
run distributed training jobs over numerous nodes in a high-performance computing
(HPC) cluster when using PyTorch DDP (Distributed Data Parallel) training.

Below is the slurm file used for COT retrieval project:
#!/bin/bash
#SBATCH --job-name=128_2cot
#SBATCH --output=128_2slurm.out
#SBATCH --error=128_2slurm.err
#SBATCH --partition=gpu2018
#SBATCH --qos=high_mem
#SBATCH --time=25:30:00
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
nvidia-smi
export CUDA_LAUNCH_BLOCKING=1
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --model_name
okamura --batch_size 128 --lr 0.01 --world_size 2

The distributed training task utilizing PyTorch DDP is really conducted with this
command. It employs torchrun, a tool for starting distributed PyTorch operations.
The —standalone and —nnodes=1 and —nproc per node=2 parameters define the
number of nodes and processes per node, The Python script that implements the training code is called main.py.
The remaining flags (—model name, —batch size, —lr, and —world size) define the
training job’s hyperparameters.

Step 7: Observe the results:
a. Check the train.slurm 
b. Check the job name 
c. Slurm.out file name 
d. Slurm.err file name 

Step 8: For the output open the slurm.out file use the following command
more slurm.out



