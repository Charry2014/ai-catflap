#!/bin/bash
# A mostly complete initial setup script
sudo apt install git python3-pip  python3-venv -y

# On the Raspberry with Bullseye, this was needed
# python3-pyqt5 
sudo apt install -y libcap-dev python3-opengl python3-picamera2 python3-libcamera python3-opencv python3-tk

# pip install --upgrade pip 
python3 -m venv venv 
source venv/bin/activate
pip3 install -r src/catflap/requirements.txt

# Hacky fix to libcamera module not found in Python, again on the Raspberry with Bullseye
cp -R /usr/lib/python3/dist-packages/libcamera/ ./venv/lib/python3.9/site-packages/
cp -R /usr/lib/python3/dist-packages/pykms/ ./venv/lib/python3.9/site-packages/

mkdir recording
mkdir log

# set the working branch to remote origin
# git reset --hard origin/master
# git init
# git remote add origin git@github.com:Charry2014/catflap.git
# git pull origin master