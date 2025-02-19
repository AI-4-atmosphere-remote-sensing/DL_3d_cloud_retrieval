# Running ML Models on AWS EC2 Instances

This guide explains how to automate running ML models on AWS EC2 instances using a Python script. The automation handles instance launching, permission configuration, model execution, log capture, S3 storage, and instance termination.

## 1. Configure AWS Credentials

Before using the script, set up your AWS credentials:

1. Install the AWS CLI if you haven't already:
   ```bash
   pip install awscli
   ```

2. Configure your AWS credentials by running:
   ```bash
   aws configure
   ```

3. You'll be prompted to enter:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region (e.g., us-west-2)
   - Default output format (e.g., json)

These credentials will be stored in `~/.aws/credentials` and `~/.aws/config` and will be used by the script to authenticate AWS API calls.

## 2. Running the Script

The script automates running ML models from an AMI. Here's how to use it:

```bash
python3 auto_ml_script.py --key_path /path/to/your-key.pem --batch_size 1024 --epochs 4 --num_of_gpu 4 --model_name cam --func MSE
```
You can change the arguments as needed while executing the ```auto_ml_scrpit```.

### Required Parameters:
- `--key_path`: Path to your EC2 key pair file (.pem)

## Optional ML Parameters:
- `--batch_size`: Batch size for training (default: 1024)
- `--epochs`: Number of training epochs (default: 1)
- `--num_of_gpu`: Number of GPUs to use (default: 3)
- `--model_name`: Model architecture (default: cam)
- `--func`: Loss function (default: MSE)

The script automatically extracts the key name from your key file path. For example, if your key path is `/path/to/my-key.pem`, it will use `my-key` as the key name.

## 3. Purpose and Workflow

This script provides end-to-end automation for machine learning tasks on AWS:

1. **Resource Provisioning**: Launches an EC2 instance from a specified AMI with the right instance type and security groups
2. **Environment Setup**: 
   - Establishes SSH connection
   - Configures AWS credentials for S3 access
   - Sets proper permissions on directories
3. **ML Execution**:
   - Runs the ML model with specified parameters
   - Streams real-time output to your local terminal
4. **Result Management**:
   - Captures complete logs
   - Extracts relevant ML outputs
   - Uploads results to S3 with a timestamped folder structure:
     ```
     s3://bucket-name/result_Joint_COT_CER_Retrievals_from_LES_clouds_YYYYMMDD_HHMMSS/
     ├── log.txt (complete terminal output)
     └── out.txt (ML-specific output)
     ```
5. **Resource Cleanup**: Automatically terminates the EC2 instance when processing completes

### Benefits

- **Cost Efficiency**: Automatically terminates instances after completion, preventing wasted resources
- **Reproducibility**: Consistent environment through AMI and parameterized execution
- **Convenience**: No need to manually SSH, run commands, or manage logs
- **Flexibility**: Configurable machine learning parameters
- **Persistence**: Results safely stored in S3 even after instance termination

# To Test, you can follow the following procedure: 

# Deep Learning for 3D Cloud Retrievals from LES cloud scences
In this GitHub repositary, you will find information on how to preprocess the raw data, how to install required packages, all the deep learning models, how to train and test the models, and how to evaluate the test results quantitively and through visualization.


## Create conda environment
```
conda create -n cot_retrieval python=3.9 
conda activate cot_retrieval
```
### Install Dependencies
* Install Pytorch: ```conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch```
* Install Torchinfo (1.7.1): ```pip install torchinfo```
* Install matplotlib (3.6.1): ```pip install matplotlib```
* Install mmcv (1.6.2): ```pip install mmcv``` [If you face error regarding mmcv when running, try ```pip install mmcv-full``` ]
* Install scikit-learn (): ```pip install scikit-learn``` [please ignore the following error message: ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. nltk 3.7 requires click, which is not installed. nltk 3.7 requires tqdm, which is not installed. ]
* Install h5py: ```pip install h5py```


### Test on single example
To test trained model by using the example data provided in this GitHub repositary:
```
python example_test.py --config configs/config_test_example.yaml
```
## Contributors
* Zahid Hassan Tushar, Department of Information Systems, University of Maryland Baltimore County, <ztushar1@umbc.edu>
* Adeleke Segun Ademakinwa, Department of Physics, University of Maryland Baltimore County, <adeleka1@umbc.edu>
* Zhibo Zhang, Department of Physics, University of Maryland Baltimore County, <zzbatmos@umbc.edu>
* Sanjay Purushotham, Department of Information Systems, University of Maryland Baltimore County, <psanjay@umbc.edu>

## References
Zahid Hassan Tushar, Adeleke Ademakinwa, Jianwu Wang, Zhibo Zhang, and Sanjay Purushotham. "Joint cloud optical thickness and cloud effective radius property retrievals using attention-based deep learning models". In AGU Annual Meeting. American Geophysical Union (AGU), 2024.

## License
  Licensed under the [MIT LICENSE](LICENSE)
