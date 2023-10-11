#!/bin/bash
# Simple script to build a new tflite model

DIR=./src/buildtflite

if [ -d "$DIR" ]; then
    cd "$DIR"
fi

if [ ! -d "./venv" ]; then
    ./setup.sh
fi

source venv/bin/activate
python3 buildtflite.py

# scp ../../test/cats.tflite pi@192.168.88.38:/home/pi/projects/catflap/
