I'll update the documentation to be more comprehensive:

# AWS ML Model Automation Script
This guide explains how to automate running ML models on AWS EC2 instances. The script automates launching an EC2 instance, executing ML code, and storing results in S3.

## Prerequisites

### 1. AWS Account Setup
1. Create an AWS account if you don't have one
2. Create IAM user with following permissions:
   - EC2 full access
   - S3 full access
   - Get your Access Key ID and Secret Access Key from IAM

### 2. Local Environment Setup
1. Install Python requirements:
   ```bash
   pip install awscli boto3 paramiko
   ```

2. Configure AWS CLI:
   ```bash
   aws configure
   ```
   Enter:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region (use us-west-2)
   - Default output format (json)

3. Create S3 bucket:
   - Create a bucket in us-west-2 region
   - Make sure bucket has public access enabled
   - Note the bucket name for script parameters

4. EC2 Key Pair:
   - Create or use existing EC2 key pair in us-west-2 region
   - Save the .pem file locally
   - Set proper permissions: `chmod 400 your-key.pem`

### 3. Code Preparation
1. Prepare your code directory structure:
   ```
   your-code/
   ├── main.py           # Main ML script
   ├── requirements.txt  # Python dependencies
   └── [other code files]
   ```

2. Zip your code:
   ```bash
   zip -r code.zip your-code/
   ```

3. Prepare data:
   ```bash
   zip -r data.zip your-data/
   ```

## Running the Script

### Basic Usage
```bash
python3 ec2_runner.py \
  --key_path /path/to/your-key.pem \
  --code_zip /path/to/code.zip \
  --data_zip /path/to/data.zip \
  --s3_bucket your-bucket-name \
  --batch_size 1024 \
  --epochs 4 \
  --num_of_gpu 4 \
  --model_name cam \
  --func MSE
```

### Required Parameters
- `--key_path`: Path to EC2 key pair file (.pem)
- `--code_zip`: Path to zipped code files
- `--data_zip`: Path to zipped data files
- `--s3_bucket`: S3 bucket name for storing results

### Optional ML Parameters
- `--batch_size`: Batch size (default: 1024)
- `--epochs`: Number of epochs (default: 1)
- `--num_of_gpu`: Number of GPUs (default: 3)
- `--model_name`: Model name (default: cam)
- `--func`: Loss function (default: MSE)

## Workflow

### 1. Instance Launch
- Launches g4dn.12xlarge instance with GPU support
- Uses AMI: ami-0339ea6f7f5408bb9 (includes CUDA and ML libraries)
- Automatically configures security group access

### 2. Code Deployment
- Uploads and extracts your code and data
- Installs requirements:
  ```
  torchinfo
  mmcv==1.6.2
  h5py
  scikit-image
  ```
- Special installation for mmcv with CUDA support

### 3. Execution
- Locates and runs main.py with specified parameters
- Streams real-time output to your terminal
- Captures all output

### 4. Results Storage
Results are stored in S3 with timestamp:
```
s3://your-bucket/result_code_YYYYMMDD_HHMMSS/
├── log.txt  # Complete execution log including setup
└── out.txt  # ML model output only
```

### 5. Cleanup
- Automatically terminates EC2 instance
- All results preserved in S3

## Troubleshooting
- Ensure AWS credentials are configured correctly
- Check S3 bucket permissions
- Verify .pem file permissions (chmod 400)
- Make sure code.zip contains main.py at the correct path
- Check EC2 instance limits in your AWS account

## Cost Management
- Script uses g4dn.12xlarge instance (costly)
- Instance terminates automatically after completion
- Monitor AWS billing dashboard
- Set up billing alerts

## Security Notes
- Keep .pem file secure
- Don't commit AWS credentials
- Use IAM roles with minimum required permissions
- Monitor AWS CloudTrail for security events

## Support
For issues:
1. Check AWS service status
2. Verify AWS credentials
3. Check script logs in S3
4. Monitor EC2 console for instance status

^^^^^^^^
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
