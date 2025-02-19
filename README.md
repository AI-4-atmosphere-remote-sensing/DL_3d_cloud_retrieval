# Running ML Models on AWS EC2 Instances

This guide explains how to automate running ML models on AWS EC2 instances using a Python script. It is designed to execute **Joint_COT_CER_Retrievals_from_LES_clouds** in the AWS Cloud using Multi GPU. The automation handles instance launching, permission configuration, model execution, log capture, S3 storage, and instance termination.

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

# Machine Learning for 3D Cloud Retrievals
In this GitHub repositary, you will find information on how to preprocess the raw data, how to install required packages, all the deep learning models, how to train and test the models, and how to evaluate the test results quantitively and through visualization. Please check out each folder for details.


## Publication
Read our preprint version of the paper [here!](https://github.com/AI-4-atmosphere-remote-sensing/DL_3d_cloud_retrieval/blob/main/Publication/CloudUNet.pdf "Go to paper link")
We will update this link when a new version comes out.

## Citing this repository
If you use the code in this repository in your own work, please cite it as follows:

Zahid Hassan Tushar, Xiangyang Meng, Adeleke Ademakinwa, Jianwu Wang, Zhibo Zhang, Sanjay Purushotham. (2023). Machine Learning for 3D Cloud Retrievals.  URL: https://github.com/AI-4-atmosphere-remote-sensing/DL_3d_cloud_retrieval/



## License
  Licensed under the [MIT LICENSE](LICENSE)
