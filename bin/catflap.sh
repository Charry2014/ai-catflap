#!/bin/bash
# Set some defaults - these are:
# To run from the current directory
# To use the streaming output
# Typical command line to run live from the camera would be:
# (venv) pi@catcampi:~/projects/catflap $ ./mousetest.sh -s 0 -w http://10.0.0.38:5000

BASE_PATH="$PWD"
STREAM=http://10.0.0.195:8000/stream.mjpg
MODEL=test/cats.tflite
WEBSITE=http://10.0.0.38:5000

# Process command line options
while getopts p:s:m:w: flag
do
    case "${flag}" in
        p) BASE_PATH=${OPTARG};;
        s) STREAM=${OPTARG};;
        m) MODEL=${OPTARG};;
        w) WEBSITE=${OPTARG};;
    esac
done

LOG_DIR="$BASE_PATH/log/"
RECORDINGS_DIR="$BASE_PATH/recordings/"

# Sort out the Python paths to find modules
pythonpathadd() {
    if [ -d "$1" ] && [[ ":$PYTHONPATH:" != *":$1:"* ]]; then
        export PYTHONPATH="${PYTHONPATH:+"$PYTHONPATH:"}$1"
    fi
}
# export PYTHONPATH="$PYTHONPATH:./modules:./modules/tflite"
pythonpathadd "./src/modules"
pythonpathadd "./src/modules/tflite"

# Create necessary directories
if [ ! -d "$RECORDINGS_DIR" ]; then
    mkdir "$RECORDINGS_DIR"
fi
if [ ! -d "$LOG_DIR" ]; then
    mkdir "$LOG_DIR"
fi

# Enter virtual environment
# this is not a perfect test, but good enough for this
if [[ "$VIRTUAL_ENV" == "" ]] 
then
  source venv/bin/activate
fi

# Always auto-restart the script if it should crash
run_command() {
python3 ./src/catflap/main.py --stream $STREAM \
                --record_path $RECORDINGS_DIR              \
                --trigger 210,180,250,280                               \
                --model $BASE_PATH/$MODEL \
                --web $WEBSITE
}

until run_command; do
    echo "Program stopped with exit code $?. Restarting" >&2
    sleep 10
done
                

