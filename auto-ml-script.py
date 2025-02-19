########################################
######## SERAJ A MOSTAFA ###############
### PhD Candiate, IS Dept. UMBC ########
########################################

#!/usr/bin/env python3
import boto3
import paramiko
import time
from datetime import datetime
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run ML model on EC2')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for the model')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--num_of_gpu', type=int, default=3, help='Number of GPUs to use')
    parser.add_argument('--model_name', type=str, default='cam', help='Model name')
    parser.add_argument('--func', type=str, default='MSE', help='Loss function')
    parser.add_argument('--key_path', type=str, required=True, 
                        help='Path to EC2 key file (.pem)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Get key name from key path (filename without extension)
    key_file_path = args.key_path
    key_name = os.path.splitext(os.path.basename(key_file_path))[0]
    
    # Configuration
    ami_id = 'ami-0e06313fd578be579'
    instance_type = 'g4dn.12xlarge'
    security_group_ids = ['sg-02524143560b47240']
    bucket_name = 'seraj-automl-storage'
    region = 'us-west-2'
    
    print(f"Using key: {key_name} (from {key_file_path})")
    
    # Initialize EC2 client
    ec2 = boto3.client('ec2', region_name=region)
    
    try:
        # 1. Launch instance
        print(f"Launching instance from AMI {ami_id}...")
        response = ec2.run_instances(
            ImageId=ami_id,
            InstanceType=instance_type,
            KeyName=key_name,
            SecurityGroupIds=security_group_ids,
            MinCount=1,
            MaxCount=1
        )
        instance_id = response['Instances'][0]['InstanceId']
        
        # 2. Wait for instance to start
        print(f"Waiting for instance {instance_id} to start...")
        waiter = ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])
        
        # 3. Get instance IP
        response = ec2.describe_instances(InstanceIds=[instance_id])
        public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
        print(f"Instance running at {public_ip}")
        
        # 4. Wait for SSH to be available
        print("Waiting for SSH to be available...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        max_attempts = 20
        connected = False
        
        # Only try to connect as ec2-user
        for attempt in range(max_attempts):
            try:
                ssh.connect(
                    public_ip,
                    username='ec2-user',
                    key_filename=key_file_path,
                    timeout=10
                )
                print("Connected as ec2-user")
                connected = True
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {str(e)}")
                time.sleep(5)
                
        # 4a. Configure AWS credentials after connecting
        print("Configuring AWS credentials...")
        session = boto3.Session()
        credentials = session.get_credentials()
        region_name = session.region_name or region
        
        if credentials:
            # Create directory and files
            stdin, stdout, stderr = ssh.exec_command("mkdir -p ~/.aws")
            stdout.channel.recv_exit_status()
            
            # Create credentials file
            cred_file_content = "[default]\n"
            cred_file_content += f"aws_access_key_id = {credentials.access_key}\n"
            cred_file_content += f"aws_secret_access_key = {credentials.secret_key}\n"
            
            if credentials.token:
                cred_file_content += f"aws_session_token = {credentials.token}\n"
                
            stdin, stdout, stderr = ssh.exec_command("cat > ~/.aws/credentials")
            stdin.write(cred_file_content)
            stdin.channel.shutdown_write()
            stdout.channel.recv_exit_status()
            
            # Create config file with region
            config_content = "[default]\n"
            config_content += f"region = {region_name}\n"
            
            stdin, stdout, stderr = ssh.exec_command("cat > ~/.aws/config")
            stdin.write(config_content)
            stdin.channel.shutdown_write()
            stdout.channel.recv_exit_status()
            
            print("AWS credentials configured")
        else:
            print("Warning: No AWS credentials found in local session")
        
        # 5. Use the exact path we know works
        joint_dir = "/home/ec2-user/Joint_COT_CER_Retrievals_from_LES_clouds"
        ml_script_path = f"{joint_dir}/main2.py"
        
        # 5a. Only fix permissions on the v71_saved_model directory
        print("Setting permissions on the log output directory...")
        log_dir = f"{joint_dir}/v71_saved_model/cam"
        
        # First create the directory with sudo if it doesn't exist
        mkdir_cmd = f"sudo mkdir -p {log_dir}"
        stdin, stdout, stderr = ssh.exec_command(mkdir_cmd)
        stdout.channel.recv_exit_status()
        
        # Then set ownership and permissions
        chown_cmd = f"sudo chown -R ec2-user:ec2-user {log_dir}"
        stdin, stdout, stderr = ssh.exec_command(chown_cmd)
        stdout.channel.recv_exit_status()
        
        chmod_cmd = f"sudo chmod -R 777 {log_dir}"
        stdin, stdout, stderr = ssh.exec_command(chmod_cmd)
        chmod_exit_status = stdout.channel.recv_exit_status()
        
        if chmod_exit_status != 0:
            print(f"Warning: Could not set permissions on log directory: {stderr.read().decode('utf-8', errors='replace')}")
        else:
            print("Log directory permissions set successfully")
        
        # 6. Create output directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"Joint_COT_CER_Retrievals_from_LES_clouds_{timestamp}"
        
        # 7. Run the ML model with arguments
        ml_cmd = f"cd {joint_dir} && python main2.py"
        
        # Add command line arguments
        if args.batch_size is not None:
            ml_cmd += f" --batch_size {args.batch_size}"
        if args.epochs is not None:
            ml_cmd += f" --epochs {args.epochs}"
        if args.num_of_gpu is not None:
            ml_cmd += f" --num_of_gpu {args.num_of_gpu}"
        if args.model_name is not None:
            ml_cmd += f" --model_name {args.model_name}"
        if args.func is not None:
            ml_cmd += f" --func {args.func}"
        
        print(f"Running ML model: {ml_cmd}")
        print("=" * 80)
        
        # Execute command
        stdin, stdout, stderr = ssh.exec_command(ml_cmd)
        
        # Stream output in real-time
        all_output = ""
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                try:
                    data = stdout.channel.recv(1024).decode('utf-8', errors='replace')
                    print(data, end='')
                    all_output += data
                except Exception as e:
                    print(f"Error reading output: {str(e)}")
            time.sleep(0.1)
            
        # Get any remaining output
        try:
            remaining = stdout.read().decode('utf-8', errors='replace')
            if remaining:
                print(remaining, end='')
                all_output += remaining
        except Exception as e:
            print(f"Error reading remaining output: {str(e)}")
            
        # Get any errors
        try:
            err = stderr.read().decode('utf-8', errors='replace')
            if err:
                print(f"STDERR: {err}")
                all_output += f"\nSTDERR: {err}"
        except Exception as e:
            print(f"Error reading stderr: {str(e)}")
            
        print("=" * 80)
        print("ML model execution completed")
        
        # 8. Save output to files
        # Get parent folder name
        parent_folder = os.path.basename(joint_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"result_{parent_folder}_{timestamp}"
        
        print(f"Saving output to S3 bucket: {bucket_name}/{output_dir}/")
        
        # Create temp directory for logs
        stdin, stdout, stderr = ssh.exec_command(f"mkdir -p /tmp/{output_dir}")
        stdout.channel.recv_exit_status()
        
        # Save all terminal output to log.txt
        echo_cmd = f"cat > /tmp/{output_dir}/log.txt"
        stdin, stdout, stderr = ssh.exec_command(echo_cmd)
        stdin.write(all_output)
        stdin.channel.shutdown_write()
        stdout.channel.recv_exit_status()
        
        # Extract ML output only for out.txt (lines related to model training/evaluation)
        ml_output = ""
        for line in all_output.split('\n'):
            if any(keyword in line for keyword in ['Epoch', 'Train:', 'Test:', 'loss', 'accuracy', 
                                                  'Running on', 'Loading', 'Batch', 'folder', 
                                                  'tensor', 'model', 'GPU']):
                ml_output += line + '\n'
        
        # Save ML output to out.txt
        echo_cmd = f"cat > /tmp/{output_dir}/out.txt"
        stdin, stdout, stderr = ssh.exec_command(echo_cmd)
        stdin.write(ml_output)
        stdin.channel.shutdown_write()
        stdout.channel.recv_exit_status()
        
        # 9. Upload to S3
        s3_upload_cmd = f"aws s3 cp /tmp/{output_dir} s3://{bucket_name}/{output_dir} --recursive"
        stdin, stdout, stderr = ssh.exec_command(s3_upload_cmd)
        exit_status = stdout.channel.recv_exit_status()
        
        if exit_status == 0:
            print(f"Logs uploaded to s3://{bucket_name}/{output_dir}/")
        else:
            error = stderr.read().decode('utf-8', errors='replace')
            print(f"Error uploading logs: {error}")
            
            # Check AWS CLI installation and configuration
            stdin, stdout, stderr = ssh.exec_command("which aws && aws --version && aws configure list")
            aws_info = stdout.read().decode('utf-8', errors='replace')
            print(f"AWS CLI info:\n{aws_info}")
            
            # Try alternative upload method
            print("Attempting alternative upload method...")
            stdin, stdout, stderr = ssh.exec_command(f"sudo yum install -y awscli && aws s3 cp /tmp/{output_dir} s3://{bucket_name}/{output_dir} --recursive")
            alt_result = stdout.read().decode('utf-8', errors='replace')
            print(f"Alternative upload result: {alt_result}")
        
        # 10. Automatically terminate the instance
        print(f"Terminating instance {instance_id}...")
        ec2.terminate_instances(InstanceIds=[instance_id])
        print("Instance termination request sent")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        
if __name__ == "__main__":
    main()
