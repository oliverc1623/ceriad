#!/bin/bash

# Install Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

# Actiavte conda
source ~/.bashrc
source ~/.zshrc

# Create conda environment for CARLA
~/miniconda3/bin/conda create -n carla python=3.8 -y

# Activate the carla conda environment
source ~/miniconda3/bin/activate carla

# Download and extract CARLA
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz -P ~/Desktop
mkdir -p ~/Desktop/CARLA_LATEST
echo "Extracting Carla to Desktop/CARLA_LATEST/"
tar -xvf ~/Desktop/CARLA_0.9.15.tar.gz -C ~/Desktop/CARLA_LATEST > /dev/null
rm ~/Desktop/CARLA_0.9.15.tar.gz

# Install Python CARLA Library
pip3 install carla

# Navigate to PythonAPI/examples directory
cd ~/Desktop/CARLA_LATEST/PythonAPI/examples

# Install python requirements
python3 -m pip install -r requirements.txt

# Finished!
echo "------------------------------------------------------------"
echo "Installation complete! Please close and re-open your console"
echo "Remember to activate your conda carla environment using \033[1mconda activate carla\033[0m every time you reopen a new console tab"

