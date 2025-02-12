########################################
######## SERAJ A MOSTAFA ###############
### PhD Candiate, IS Dept. UMBC ########
########################################

import boto3
import paramiko
import time
import datetime
import os

# AWS and EC2 Configurations
image_id = 'ami-0e7dd065af02cf477'
instance_type = 'g4dn.12xlarge'
key_name = 'seraj-oct24'
security_group_ids = ['sg-02524143560b47240']
key_file_path = '/Users/s172/Documents/ec2/seraj-oct24.pem'
s3_bucket = 'seraj-automl-storage'

# Command execution details
remote_dir = "/home/ec2-user/Joint_COT_CER_Retrievals_from_LES_clouds/"
command = "python main-gpu.py --batch_size 1024 --epochs 2 --num_of_gpu 1 --model_name cam --func MSE"

# Generate a unique timestamp for logging
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
s3_folder = f"{timestamp}/"
s3_output_file = f"{s3_folder}out.txt"

# Initialize AWS Clients
ec2 = boto3.resource('ec2')
s3_client = boto3.client('s3')

# Launch EC2 Instance
instance = ec2.create_instances(
    ImageId=image_id,
    InstanceType=instance_type,
    KeyName=key_name,
    SecurityGroupIds=security_group_ids,
    MinCount=1,
    MaxCount=1,
)[0]
print(f"Instance {instance.id} is launching...")
instance.wait_until_running()
instance.reload()
public_ip = instance.public_ip_address
print(f"Instance {instance.id} is running at {public_ip}")

# Wait a bit for SSH to be ready
time.sleep(60)

# SSH into the Instance
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(public_ip, username='ec2-user', key_filename=key_file_path)

# Execute the command
stdin, stdout, stderr = ssh.exec_command(f"cd {remote_dir} && {command}")

# Stream and store logs
log_data = ""
error_data = ""
while True:
    line = stdout.readline()
    if not line:
        break
    print(line, end="")  # Print to local terminal
    log_data += line

while True:
    err_line = stderr.readline()
    if not err_line:
        break
    print(err_line, end="")  # Print to local terminal
    error_data += err_line

# Ensure command execution is complete
stdout.channel.recv_exit_status()

# Combine stdout and stderr logs
full_log_data = log_data + "\nERROR LOGS:\n" + error_data

# Upload logs to S3
s3_client.put_object(Bucket=s3_bucket, Key=s3_output_file, Body=full_log_data.encode('utf-8'))
print(f"Logs uploaded to s3://{s3_bucket}/{s3_output_file}")

# Keep the instance running for manual inspection if needed
print("Execution completed. Instance will NOT be terminated automatically.")
