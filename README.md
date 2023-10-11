# catflap
An AI based mouse detecting cat flap extension for the Sure Flap cat flap. Inspired by a [project by Gerrit Bojen](https://towardsdatascience.com/keep-your-home-mouse-free-with-an-ai-powered-cat-flap-a67c686ce394).


# Introduction

During the COVID pandemic we got two cats as a way of distracting the children from the horrors of being locked up at home together. The cats, being cats, set about discovering the neighbourhood and then filling our house with all manner of furry critters. These poor animals are unfortunate trophies of their hunting ability, to show us how much they love us, or to help us hone our own hunting abilities - depending what you believe about cats' motivations. Anyway, we did not find any of these options particularly appealing - and certainly not practicing our hunting ability in our own living room. 

Our cats are good hunters. We have had many, many mice, mice of all kinds - big and small, small rats, voles, even a baby sand martin. Some have been very much alive - running around the house, eating the cats' food, climbing the curtains, making nests under the sofa - and we have indeed honed our ability to catch and eject these. We are first aware of the presence of a furry guest when we see the cats demonstrating hunting behaviours around the house, then we step in. Worse, however, are the ones alive enough to escape the cat, but then die under the sofa - you can imagine what happens next.

Given this situation we had to do something. Being a pragmatist the first step was to restrict the cat flap to exit-only, but being an engineer I set about providing a better solution to the problem.

# Getting Started
## Background

This has been a long project with much learning along the way, and while everyone here wants to get down to the cool machine learning stuff remember that you to spend time on thinking through the environment in which this will work. These instructions have become long because this is not just a machine learning problem - to succeed you must master many things. You will have a camera, making videos of your cats, in their natural behaviours, and getting a clean, well exposed shot of your cat for the ML to work on is paramount. Avoiding false triggers from other background motion is really helpful. Our cats come in mostly at night so the Pi is using its IR illumination, but they also want to get in in daylight. This needed a wooden construction around the cat flap to guide the cats' approach so they can pose for the camera.

Here is the construction around the cat flap to guide the cats past the camera:
![20230224_094731](https://user-images.githubusercontent.com/58067238/221138634-3a96eb61-15f4-4785-8fe3-4012f59f9fb6.jpg)

And here is the Pi in its weather-resistant box:
![20230224_094748](https://user-images.githubusercontent.com/58067238/221138742-7d43f4b6-a6c0-4874-99c4-a8b0db862b1b.jpg)

The sun comes round and in certain times of the year blinds the camera. All these things complicate the cool stuff. At first the video was streaming over wifi but despite a high performance Unifi Wifi setup, strong signal and careful setup it was still unreliable so now it uses a LAN cable to the Pi.

Then getting all the links in the long chain from cat-posing-for-camera to functioning machine learning model required a good deal of fiddling. These instructions might help you get there faster.

## Architecture Overview

As good as the Raspberry Pi has become it is still more comfortable to develop on the desktop (MacBook Pro or various Debian Linux flavours, in my case) so the Camera Pi was set up in the beginning to stream video into the LAN. This meant that we could watch the cats come and go from our computers & phones and add a video window to the Home Assistant dashboard in the kitchen. This helped create interest in the family and it was no longer just 'one of Daddy's projects' but something for 'all of us'. It was also very helpful to have the video streaming to the phone while adjusting the camera and getting it pointing exactly right. Having studied the cats' approach paths it was just a matter of pointing the camera, and restricting some approaches that avoided the camera.

For any ML project the first step is data collection - so this means motion detection, recording of videos, extraction of still images, classification of the stills, and training a model.

![image](https://user-images.githubusercontent.com/58067238/221131208-d5e5af75-dab2-4e0f-b912-3472b6458bc8.png)

Given the working model then it is deployment onto the Pi and connecting to a GPIO and some electronics to control the cat flap.

# Initial Development Stages

There is a lot to cover here so I will try to break it down into some logical elements. These parts are critical to making this work - think through each stage and decide what will work for you.

## Basic Setup

You will want to make sure the Pi is mounted in the right place so the camera can see the cats, they need to be approaching the camera for a few seconds so you get some good clean shots of the cat's face, well exposed, and of course you want to make sure the Pi is protected from the weather, and has a wired LAN connection.

## Configure the Pi
Using the Raspberry Pi 4 with 4GB RAM. The Pi 4 is basically impossible to buy now in 2023 unless you pay a scalper 3x what it should cost, but the Orange Pi 5 looks like a good alternative - it also has an ARM Mali-G610 GPU as well as a 6 TOPS NPU and of course the A55 quad core CPU so it might be much faster, anyway, I digress. You need a Pi for this.

The Pi in this example have stock [Raspberry Pi OS](https://www.raspberrypi.com/software/) installed with the GUI, and have been configured to allow ssh and VNC access - see `Preferences` -> `Raspberry Pi Configuration` under the Raspberry menu. The Pi with the camera on it will also need the camera enabled in the same place. The Pi needs the GUI so you can display images for orienting the camera and tuning the motion detection target area later. While technically you could run all these servers headless in practice it turned out to be a real pain to do this.
![image](https://user-images.githubusercontent.com/58067238/221145036-22bf5258-621f-4ddc-a38c-2d5bb498351f.png)

The Pi have SSH keys generated with `ssh-keygen` and then used for enabling [key based SSH login](https://www.geekyhacker.com/2021/02/15/configure-ssh-key-based-authentication-on-raspberry-pi/) to the Pi and for GitHub. I do this for convenience in my LAN, not for security.

This project is created with [Visual Studio Code](https://code.visualstudio.com/) and the Remote - SSH(https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) plugin is very comfortable for debugging on the Pi. From the command palette you can then Add Remote host and enter `<pi_username>@<your.pi.ipaddress>`

## Python & Virtual Environments

Python is a wonderful language but not without its problems. This project does not use the latest, greatest Python because it is necessary to make sure that the main modules - TensorFlow Lite and OpenCV among them - are compatible. Generally I use [Python 3.9.10](https://www.python.org/downloads/release/python-3910/). Everyhwere Python is used in this project it is running in a virtual environment - this is good practice for Python.

[Install OpenCV for the Raspberry Pi](https://raspberrypi-guide.github.io/programming/install-opencv)


## Parts List

* Raspberry Pi 4 with 4GB RAM and 32GB SD card
* Raspberry Pi Camera with IR illumination
* Box – BOPLA M 220 G Gehäuse Serie Euromas I, 160 x 80 x 55 mm, IP65
* PCB Spacers for Raspberry Pi
*  4x M2.5 Abstandsbolzen, Länge 20 mm
*  4x M2.5 4 mm Schraube
*  4x Mutter

## Mount the Pi and Start Streaming

Credits here go to the [Pi standard streaming example](http://picamera.readthedocs.io/en/latest/recipes2.html#web-streaming) - detailed instructions on [Random Nerd Tutorials](https://randomnerdtutorials.com/video-streaming-with-raspberry-pi-camera/).
Connect to the camera Pi with VNC and start the streaming with:
```
mkdir catflap
cd catflap
git clone git@github.com:Charry2014/catflap.git
cd streamer
python3 pi_streamer.py
```
Standard resolution is 640 x 480. With the Pi streaming you can start to orientate the Pi correctly. Viewing the video stream on your phone in a browser is easy on `http://<your.pi.ipaddress>:8000/stream.mjpg`. The camera is actually 1080p but we do not need all those extra pixels.

## Set up The Motion Detection & Recorder
In my example I am recording to the local storage on the Pi and then using a cronjob to copy these every hour to my NAS. This is because I already have my NAS mounted everywhere in the network and it is then easier to review the recorded videos than having the videos pile up on the Pi. Make your own choices here - but remember you will frequently need to watch and filter the videos so making this easy is a good idea.

The recorder in this example is running on a Ubuntu Linux 20.04 LTS server - as before clone the repository into a `./catflap` directory and then this should get you set up:
```
cd motiondetect
./setup.sh
python3 motiondetect.py --stream http://<your.pi.ipaddress>:8000/stream.mjpg --record_path ./recording --trigger <x,y,w,h> --show_trigger --record_overlays
```
The `x,y,w,h` coordinates to the `--trigger` option define the area used to trigger the motion detection so you can hone this to avoid false triggers from other moving items such as plants blowing in the wind or rain, snow, other flying creatures etc. There is much that can be done to refine the trigger and further avoid false positives but for now we have this simple area. 

**Note** - The options `--show_trigger` and `--record_overlays` are good for initial tuning but you will want to remove these when you start collecting real videos for ML training.

Use a cronjob to copy the recordings `@hourly /path/to/catflap/copy_recordings.sh`.

# Data Collection & Labelling
This is the start of the good stuff - you now have your Pi set up, camera streaming, detecting motion and recording videos to a sensible place. The cats, when they show up in these videos, are well exposed at least for a few seconds and you get at least for a brief moment a nice shot of the cat's head. 

It is easy to get into a big mess handling loads of videos, still images taken from these, and keeping 'new' videos from those you have already labelled.


## Machine Learning Basics

It is helpful to define early on the categories that we will separate videos into as this feeds through a lot of decisions that have to be made when handling recordings and throughout the machine learning process. 

For mouse detection let's choose:
* `Cat-alone` - there is a cat without a mouse
* `Cat-with-mouse` - there is a cat who is carrying a mouse
* `Background` - nothing to see here

When I review the recorded videos I delete the irrelevant ones then sort them into folders `./incoming/cat-alone`, `./incoming/cat-with-mouse` etc. so I know these are new incoming videos and then I can label them in a organised fashion.

In ML terms there are two approaches that might be interesting to try for mouse detection - these are *Image Classification* and *Object Detection*. There are many posts on the 'net about this but it is enough to understand that Image Classification just aims to say what the image contains - ie. picture of dog, picture of cat, picture of cat with mouse, etc. Object detection goes a level deeper and attempts to find where on the image you can see any dog, cat, banana, whatever. This project uses an Image Classification approach. There are also many frameworks available and [Q-Engineering](https://qengineering.eu/deep-learning-software-for-raspberry-pi-and-alternatives.html) have a nice summary of these. This project just uses Google's TensorFlow Lite.

## Getting Started

* Install [VLC](https://www.videolan.org/vlc/) - follow the instructions on the site
* Install [Label Studio](https://labelstud.io/) in the Python virtual environment. Command lines using `pip` given below.

```
sudo apt install git python3-pip  python03-venv -y
python3 -m venv venv 
source venv/bin/activate
pip install -U label-studio
label-studio &
```

## Still Image Extraction

Next we filter videos into - `Cat-alone`, `Cat-with-mouse`, `Background` categories. Here I make a folder structure  `./<date>/cat-alone`, `./<date>/cat-with-mouse` and I move as many of the videos from `incoming` as I have time to label into this folder. This way it is clear which videos have been processed.
Start up VLC and configure to export JPEG files jpg into the folders above. TensorFlow Lite does not accept PNG so JPG is important - I did not know this and labelled a bunch of images JPG - doh! Go to VLC -> `Preferences` -> `Video tab` ->  and set the folder to somewhere sensible, the format to `JPEG`, the prefix to `$N-` so the original video name is retained in the snapshot filename and tick the `sequential numbering` box.

image.png


## Labelling

Image Studio
Import images

## Building a TensorFlow Lite Model

Export from Label Studio as Pascal VOC format 
Run the Python script to modify the XML and convert .png to jpg.


# Testing the ML Model

## Installing on the Cat Cam Pi

https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi



```
cd mousetest

sudo apt-get install libatlas-base-dev
sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 python3-pyqt5 python3-dev -y


python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt

python3 motiondetect.py 
        --trigger 250,180,350,280
        --show_trigger
        --record_overlays
```




# Modifying the Cat Flap
-- work in progress --

[Sure Flap cat flap disassembly instructions on the FCC website](https://fccid.io/XO9-FLAP-1001/Internal-Photos/INTERNAL-PHOTOS-1238385). If you have never used the FCC website it is sometimes very helpful for finding out the internal details of many products.


