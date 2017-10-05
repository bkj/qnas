#!/bin/bash

# provision-qnas-ec2.sh

sudo apt-get update
sudo apt-get install -y git wget

# --
# Install CUDA

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install -y --assume-yes cuda
rm cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
echo 'export LIBRARY_PATH="/usr/local/cuda/lib64/:$LIBRARY_PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64/:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc

scp -i .ssh/rzj.pem ubuntu@54.147.122.74:/home/ubuntu/bl-resources/cuda/cudnn-8.0-linux-x64-v6.0.tgz ./
tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz
sudo mv cuda/include/* /usr/local/cuda/include/
sudo mv cuda/lib64/* /usr/local/cuda/lib64/
rm -rf cuda

# --
# Install docker

sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update
sudo apt-get install -y docker-ce

# --
# Install nvidia-docker

wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

# --
# Fetch qnas

mkdir -p projects
cd projects
git clone https://github.com/bkj/qnas -b python3
cd qnas/docker
sudo docker build -t qnas3 .

