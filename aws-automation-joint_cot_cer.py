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
    # EC2 and AWS configuration
    parser.add_argument('--key_path', type=str, required=True, 
                        help='Path to EC2 key file (.pem)')
    parser.add_argument('--s3_bucket', type=str, required=True,
                        help='S3 bucket name for storing results')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Local directory path to save log files')
    
    # File paths
    parser.add_argument('--code_zip', type=str, required=True,
                        help='Path to zip file containing code')
    parser.add_argument('--data_zip', type=str, required=True,
                        help='Path to zip file containing data')
    
    # ML parameters
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_of_gpu', type=int, default=3)
    parser.add_argument('--model_name', type=str, default='cam')
    parser.add_argument('--func', type=str, default='MSE')
    return parser.parse_args()

def run_command(ssh, command, print_output=True):
    """Helper function to run commands and handle output"""
    stdin, stdout, stderr = ssh.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    if print_output:
        output = stdout.read().decode('utf-8', errors='replace')
        error = stderr.read().decode('utf-8', errors='replace')
        if output:
            print(output)
        if error:
            print(f"Error: {error}")
    return exit_status

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize terminal output capture
    terminal_logs = []
    
    def log_print(message):
        """Helper to print and store output"""
        print(message)
        terminal_logs.append(message)
    
    # Get key name from key path (filename without extension)
    key_file_path = args.key_path
    key_name = os.path.splitext(os.path.basename(key_file_path))[0]
    
    # Configuration
    ami_id = 'ami-0339ea6f7f5408bb9'
    instance_type = 'g5.12xlarge'
    security_group_ids = ['sg-02524143560b47240']
    region = 'us-west-2'
    
    log_print(f"Using key: {key_name} (from {key_file_path})")
    log_print(f"Using S3 bucket: {args.s3_bucket}")
    
    try:
        # Initialize EC2 client
        ec2 = boto3.client('ec2', region_name=region)
        
        # 1. Launch instance
        log_print(f"Launching instance from AMI {ami_id}...")
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
        log_print(f"Waiting for instance {instance_id} to start...")
        waiter = ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])
        
        # 3. Get instance IP and DNS
        response = ec2.describe_instances(InstanceIds=[instance_id])
        instance_info = response['Reservations'][0]['Instances'][0]
        public_ip = instance_info['PublicIpAddress']
        public_dns = instance_info['PublicDnsName']
        log_print(f"Instance running at:")
        log_print(f"IP: {public_ip}")
        log_print(f"DNS: {public_dns}")
        log_print(f"SSH command: ssh -i {key_file_path} ec2-user@{public_dns}")
        
        # 4. Wait for SSH to be available
        log_print("\nWaiting for SSH to be available...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        max_attempts = 20
        connected = False
        username = 'ec2-user'  # Try ec2-user first
        
        # Try to connect using DNS name
        for attempt in range(max_attempts):
            try:
                ssh.connect(
                    public_dns,  # Use DNS instead of IP
                    username=username,
                    key_filename=key_file_path,
                    timeout=10
                )
                log_print(f"Connected as {username} to {public_dns}")
                connected = True
                break
            except Exception as e:
                log_print(f"Attempt {attempt+1} failed: {str(e)}")
                if username == 'ec2-user' and attempt == 2:
                    log_print("Trying ubuntu user instead...")
                    username = 'ubuntu'
                time.sleep(5)
                
        if not connected:
            raise Exception("Failed to connect after multiple attempts")

        # 5. Configure AWS credentials
        log_print("Configuring AWS credentials...")
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials:
            stdin, stdout, stderr = ssh.exec_command("mkdir -p ~/.aws")
            stdout.channel.recv_exit_status()
            
            cred_file_content = "[default]\n"
            cred_file_content += f"aws_access_key_id = {credentials.access_key}\n"
            cred_file_content += f"aws_secret_access_key = {credentials.secret_key}\n"
            if credentials.token:
                cred_file_content += f"aws_session_token = {credentials.token}\n"
                
            stdin, stdout, stderr = ssh.exec_command("cat > ~/.aws/credentials")
            stdin.write(cred_file_content)
            stdin.channel.shutdown_write()
            stdout.channel.recv_exit_status()
            
            config_content = "[default]\n"
            config_content += f"region = {region}\n"
            
            stdin, stdout, stderr = ssh.exec_command("cat > ~/.aws/config")
            stdin.write(config_content)
            stdin.channel.shutdown_write()
            stdout.channel.recv_exit_status()

        # 6. Upload and extract code and data
        log_print("Uploading code and data files...")
        sftp = ssh.open_sftp()
        
        # Set home directory based on user
        home_dir = f"/home/{username}"
        
        # Upload code
        remote_code_zip = f"{home_dir}/code.zip"
        sftp.put(args.code_zip, remote_code_zip)
        
        # Upload data
        remote_data_zip = f"{home_dir}/data.zip"
        sftp.put(args.data_zip, remote_data_zip)
        
        sftp.close()
        
        # Extract files
        log_print("Extracting files...")
        run_command(ssh, f"unzip -o {remote_code_zip} -d {home_dir}/")
        run_command(ssh, f"unzip -o {remote_data_zip} -d {home_dir}/")
        
        # Find main.py location
        log_print("Locating main.py...")
        stdin, stdout, stderr = ssh.exec_command(f"find {home_dir} -name 'main.py'")
        main_py_path = stdout.read().decode('utf-8').strip()
        if not main_py_path:
            raise Exception("Could not find main.py in the uploaded code")
            
        working_dir = os.path.dirname(main_py_path)
        log_print(f"Found main.py in: {working_dir}")
        
        # 7. Install required packages
        log_print("Installing required packages...")
        run_command(ssh, "pip install torchinfo h5py scikit-image")
        run_command(ssh, "pip install mmcv==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html")
        
        # Wait for installations to complete
        log_print("Waiting for installations to complete...")
        time.sleep(4)
        
        # 8. Run the ML model
        ml_cmd = f"cd {working_dir} && python main.py"
        
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
            
        log_print(f"Running ML model: {ml_cmd}")
        log_print("=" * 80)
        
        # Execute command and stream output
        stdin, stdout, stderr = ssh.exec_command(ml_cmd)
        
        all_output = ""
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                try:
                    data = stdout.channel.recv(1024).decode('utf-8', errors='replace')
                    print(data, end='')
                    all_output += data
                except Exception as e:
                    log_print(f"Error reading output: {str(e)}")
            time.sleep(0.1)
            
        try:
            remaining = stdout.read().decode('utf-8', errors='replace')
            if remaining:
                print(remaining, end='')
                all_output += remaining
        except Exception as e:
            log_print(f"Error reading remaining output: {str(e)}")
            
        try:
            err = stderr.read().decode('utf-8', errors='replace')
            if err:
                print(f"STDERR: {err}")
                all_output += f"\nSTDERR: {err}"
        except Exception as e:
            log_print(f"Error reading stderr: {str(e)}")
            
        print("=" * 80)
        log_print("Model execution completed")
        
        # 9. Save output locally and to S3
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"result_{os.path.basename(working_dir)}_{timestamp}"
        log_print(f"Saving output to S3 bucket: {args.s3_bucket}/{output_dir}/")
        
        # Create local directory
        local_output_path = os.path.join(args.output_dir, output_dir)
        os.makedirs(local_output_path, exist_ok=True)
        
        # Create full log content
        full_log = "\n".join(terminal_logs)
        full_log += "\n\nML Output:\n" + all_output
        
        # Save locally first
        with open(os.path.join(local_output_path, 'log.txt'), 'w') as f:
            f.write(full_log)
        with open(os.path.join(local_output_path, 'out.txt'), 'w') as f:
            f.write(all_output)
        log_print(f"Logs saved locally in directory: {local_output_path}/")
        
        # Create temp directory for S3 upload
        stdin, stdout, stderr = ssh.exec_command(f"mkdir -p /tmp/{output_dir}")
        stdout.channel.recv_exit_status()
        
        # Write files for S3 upload
        echo_cmd = f"cat > /tmp/{output_dir}/log.txt"
        stdin, stdout, stderr = ssh.exec_command(echo_cmd)
        stdin.write(full_log)
        stdin.channel.shutdown_write()
        stdout.channel.recv_exit_status()
        
        echo_cmd = f"cat > /tmp/{output_dir}/out.txt"
        stdin, stdout, stderr = ssh.exec_command(echo_cmd)
        stdin.write(all_output)
        stdin.channel.shutdown_write()
        stdout.channel.recv_exit_status()
        
        # Upload to S3
        s3_upload_cmd = f"aws s3 cp /tmp/{output_dir} s3://{args.s3_bucket}/{output_dir} --recursive"
        stdin, stdout, stderr = ssh.exec_command(s3_upload_cmd)
        exit_status = stdout.channel.recv_exit_status()
        
        if exit_status == 0:
            log_print(f"Logs uploaded to s3://{args.s3_bucket}/{output_dir}/")
        else:
            error = stderr.read().decode('utf-8', errors='replace')
            log_print(f"Error uploading logs: {error}")
        
        # 10. Terminate instance
        log_print(f"Terminating instance {instance_id}...")
        ec2.terminate_instances(InstanceIds=[instance_id])
        log_print("Instance termination request sent")
            
    except Exception as e:
        log_print(f"Error: {str(e)}")
        
if __name__ == "__main__":
    main()
