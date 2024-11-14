### MULTI GPU based Cloud Optical Thickness (COT) Retrievals.

# Guide to Run `script.py`

This guide explains how to set up your AWS environment and run `script.py` for executing tasks on an EC2 instance with multiple GPUs.

## AWS Configuration Steps

### 1. Configure AWS CLI

When prompted, enter the following:

- **AWS Access Key ID**: Enter your AWS Access Key ID.
- **AWS Secret Access Key**: Enter your AWS Secret Access Key.
- **Default region name**: Enter `us-west-2` or your preferred region.
- **Default output format**: Enter `json`, or choose `yaml`, `text` if desired.

### 2. Required Permissions

Ensure your AWS user has the following permissions:

- **EC2**: Full access to launch, manage, and terminate instances.
- **S3**: Read and write permissions for data storage.
- **IAM**: Permissions to manage roles and instance profiles.
- **CloudWatch (Optional)**: For monitoring logs and metrics.

### 3. Gather Required AWS Details

You need the following details to launch an EC2 instance:

- **AMI ID**: Example: `ami-05c456ebf5c525b7b`
- **Subnet ID**: Example: `subnet-5e38fc14`
- **Security Group ID**: Example: `sg-02524143560b47240`
- **Key Pair**: Ensure you have access to the `.pem` file (e.g., `your-key-pair.pem`) for SSH.

### 4. Instance Type

Select an instance type with multiple GPUs. Recommended: `g4dn.12xlarge` or similar for high GPU performance.

## Running the Script

Once you have configured your AWS environment and gathered the required information, you can run `script.py`.

### Steps to Execute

**Run the script:**

```python script.py```
