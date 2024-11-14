# Use the official Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install PyTorch for CPU
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other Python packages using pip
RUN pip install numpy scipy matplotlib scikit-learn h5py mmcv==1.5.3 torchinfo

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code into the container
COPY . .

# Set the default command for the container
CMD ["python", "test.py", "--model_arch", "cloudunet", "--model_filename", "cloudunet.pth"]
