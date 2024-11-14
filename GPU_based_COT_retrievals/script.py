import boto3
import paramiko
import time
import os
import sys

class EC2ModelLauncher:
	def __init__(self, 
				 ami_id='ami-0xxxxxxxxx', ##update 
				 instance_type='g4dn.12xlarge', ## change as necessary
				 security_group_id='sg-0252xxxxxxx', ## update
				 vpc_id='vpc-8xxxxxx', ## update
				 key_file_path='/path/to/your.pem'):
		self.ec2 = boto3.client('ec2')
		
		self.ami_id = ami_id
		self.instance_type = instance_type
		self.security_group_id = security_group_id
		self.vpc_id = vpc_id
		self.key_file_path = key_file_path
		self.key_name = os.path.splitext(os.path.basename(key_file_path))[0]
		
	def launch_instance(self):
		try:
			response = self.ec2.run_instances(
				ImageId=self.ami_id,
				InstanceType=self.instance_type,
				KeyName=self.key_name,
				SecurityGroupIds=[self.security_group_id],
				MinCount=1,
				MaxCount=1,
				SubnetId=self.get_subnet_id()
			)
			
			instance_id = response['Instances'][0]['InstanceId']
			
			waiter = self.ec2.get_waiter('instance_running')
			waiter.wait(InstanceIds=[instance_id])
			
			instance_info = self.ec2.describe_instances(InstanceIds=[instance_id])
			public_ip = instance_info['Reservations'][0]['Instances'][0]['PublicIpAddress']
			
			print(f"Instance launched successfully. Instance ID: {instance_id}")
			print(f"Public IP: {public_ip}")
			
			return instance_id, public_ip
		
		except Exception as e:
			print(f"Error launching instance: {e}")
			return None, None
	
	def get_subnet_id(self):
		ec2_resource = boto3.resource('ec2')
		vpc = ec2_resource.Vpc(self.vpc_id)
		
		for subnet in vpc.subnets.all():
			return subnet.id
		
		raise Exception("No subnets found in the specified VPC")
	
	def connect_and_run_model(self, public_ip):
		ssh = paramiko.SSHClient()
		ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		
		try:
			# Connect to the instance
			ssh.connect(
				hostname=public_ip, 
				username='ec2-user',  # Adjust if different
				key_filename=self.key_file_path
			)
			
			# Combine commands into a single SSH session
			full_command = ' && '.join([
				'cd /home/ec2-user',
				'cd COT_retrievals_from_LES_cloud_scenes_reflectances/',
				# Alternative log path
				'python main.py --batch_size 64 --epochs 2 --num_of_gpu 1 --lr 0.1 --model_name cloudunet --log_dir /tmp/ml_logs/cloudunet'    
			])            
			print("Executing full command:", full_command)
			
			# Execute command and stream output in real-time
			stdin, stdout, stderr = ssh.exec_command(full_command)
			
			# Stream stdout in real-time
			print("\n--- STDOUT ---")
			for line in stdout:
				print(line.strip())
			
			# Stream stderr in real-time
			print("\n--- STDERR ---")
			for line in stderr:
				print(line.strip())
			
			# Check exit status
			exit_status = stdout.channel.recv_exit_status()
			print(f"\nCommand exit status: {exit_status}")
			
			return exit_status == 0  # Return True if successful
		
		except Exception as e:
			print(f"Error connecting or running commands: {e}")
			return False
		
		finally:
			ssh.close()
	
	def terminate_instance(self, instance_id):
		try:
			self.ec2.terminate_instances(InstanceIds=[instance_id])
			print(f"Instance {instance_id} terminated successfully")
		except Exception as e:
			print(f"Error terminating instance: {e}")
			# Attempt to force termination
			try:
				self.ec2.terminate_instances(InstanceIds=[instance_id], Force=True)
				print(f"Forced termination of instance {instance_id}")
			except Exception as force_e:
				print(f"Failed to force terminate instance: {force_e}")
				# Log critical error
				with open('ec2_termination_error.log', 'w') as f:
					f.write(f"Critical: Could not terminate instance {instance_id}\n")
					f.write(str(force_e))
	
	def run_ml_workflow(self):
		instance_id = None
		try:
			# Launch instance
			instance_id, public_ip = self.launch_instance()
			
			if instance_id and public_ip:
				# Wait a bit for instance to fully initialize
				print("Waiting for instance to initialize...")
				time.sleep(120)
				
				# Run ML model
				success = self.connect_and_run_model(public_ip)
				
				# Always terminate the instance
				self.terminate_instance(instance_id)
				
				# Exit with appropriate status
				sys.exit(0 if success else 1)
		
		except Exception as e:
			print(f"Workflow failed: {e}")
			
			# Ensure instance is terminated even if something goes wrong
			if instance_id:
				self.terminate_instance(instance_id)
			
			sys.exit(1)

# Usage
if __name__ == "__main__":
	launcher = EC2ModelLauncher()
	launcher.run_ml_workflow()
