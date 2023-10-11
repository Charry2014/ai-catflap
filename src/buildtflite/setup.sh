#!/bin/bash
# A simple initial setup script - creates a venv and installs the necessary stuff
# If needed you need to install pip and venv
# sudo apt install python3-pip  python3-venv -y
if [ -d "venv" ]; then
    rm -rf venv
fi
python3 -m venv venv 
source venv/bin/activate
pip3 install -r requirements.txt

# Run the build with
# python3 ./buildtflite.py