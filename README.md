# catflap
An AI based mouse detecting cat flap extension for the Sure Flap cat flap. Inspired by a [project by Gerrit Bojen](https://towardsdatascience.com/keep-your-home-mouse-free-with-an-ai-powered-cat-flap-a67c686ce394). It includes collecting data, training a model, deploying to the Pi, monitoring performance.

Describing this as an AI project is a bit like saying that driving a car is just a matter of turning the wheel. There are many skills to master and despite the plentiful YouTube videos showing you ML on the Pi in 10 minutes, this has been a long project with many problems overcome. The machine learning part of this is by far not the most difficult, thanks to the work of those wonderful people at Google - but getting to that point is hard.

# Introduction

During the COVID pandemic we got two cats as a way of distracting the children from the horrors of being locked up at home together. The cats, being cats, set about discovering the neighbourhood and then filling our house with all manner of furry critters. These poor animals are unfortunate trophies of their hunting ability, to show us how much they love us, or to help us hone our own hunting abilities - depending what you believe about cats' motivations. Anyway, we did not find any of these options particularly appealing - and certainly not practicing our hunting ability in our own living room. 

Our cats are good hunters. We have had many furry critters of all kinds - big and small, small rats, voles, even a baby sand martin. Some have been very much alive - running around the house, eating the cats' food, climbing the curtains, making nests under the sofa - and we have indeed honed our ability to catch and eject these. We are first aware of the presence of a furry guest when we see the cats demonstrating hunting behaviours around the house, then we step in. Worse, however, are the ones alive enough to escape the cat, but then die under the sofa - you can imagine what happens next.

Given this situation we had to do something. Being a pragmatist the first step was to restrict the cat flap to exit-only, but being an engineer I set about providing a better solution to the problem.

# Getting Started
## Background

Here, in approximate sequence of necessity, are those stages.

## Setup a Development Environment

As good as the Pi OS is, it is still more comfortable to develop on a proper desktop system - I use either a Debian Linux (Ubuntu) desktop or macOS for this. Here comes the first gotcha - this is now **cross-platform development**. not every version of Python is easily installable on the Pi, so you will want to use the same version on the desktop as available on the Pi - currently Python 3.9.10.

Then for an IDE - I use [Visual Studio Code](https://code.visualstudio.com/) and the [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) plugin is very comfortable for debugging on the Pi. From the command palette you can then Add Remote host and enter `<pi_username>@<your.pi.ipaddress>`. The Pi have SSH keys generated with `ssh-keygen` and then used for enabling [key based SSH login](https://www.geekyhacker.com/2021/02/15/configure-ssh-key-based-authentication-on-raspberry-pi/) to the Pi and for GitHub.

Oh, and of course, only develop Python inside a virtual environment.

## Data Collection

Without data there is no machine learning. I set up a Pi with a camera and added some motion detection software to it. This was cobbled together using the Raspberry Pi example for streaming from a camera, and the OpenCV example for motion detection. The code here is so ugly I will share it only when I have cleaned it up. The stream was very useful to be able to monitor the camera live from anywhere on my LAN, very handy for early understanding how the camera siting was working, or not.

Credits here go to the [Pi standard streaming example](http://picamera.readthedocs.io/en/latest/recipes2.html#web-streaming) - detailed instructions on [Random Nerd Tutorials](https://randomnerdtutorials.com/video-streaming-with-raspberry-pi-camera/). Note - with the switch to Bullseye the operation of the Pi camera changed completely.

On to the stream was a client that detected motion and recorded a 10 second video every time motion was detected. From here the first set of images were collected.

You will want to make sure the Pi is mounted in the right place so the camera can see the cats, they need to be approaching the camera for a few seconds so you get some good clean shots of the cat's face, well exposed, and of course you want to make sure the Pi is protected from the weather, and has a wired LAN connection.

## Image Labelling - Preparation

From the videos a series of still images was captured using VLC

* Install [VLC](https://www.videolan.org/vlc/) - follow the instructions on the site

Start up VLC and configure to export JPEG files jpg into the folders above. TensorFlow Lite does not accept PNG so JPG is important! Go to VLC -> `Preferences` -> `Video tab` ->  and set the folder to somewhere sensible, the format to `JPEG`, the prefix to `$N-` so the original video name is retained in the snapshot filename and tick the `sequential numbering` box.

The following labels are used - `Cat-alone`, `Cat-with-mouse`, `Cat-body`, `Background` categories. Collect as many images as your patience allows in all categories and add them to the labelling tool.

The labels explained -
* `Cat-alone` - The cat's mouth is clearly visible and there is nothing in it
* `Cat-with-mouse` - The cat's mouth is clearly visible and there is a mouse there
* `Cat-body` - Some other part of the cat is visible, not the mouse
* `Background` - No part of a cat is visible, this is a false trigger and ignored in subsequent processing

## Image Labelling - Label Studio

* Install [Label Studio](https://labelstud.io/) in the Python virtual environment. Command lines using `pip` given below.

```
sudo apt install git python3-pip  python03-venv -y
python3 -m venv venv 
source venv/bin/activate
pip install -U label-studio
label-studio &
```

Set up a project, add the labels, and import your images. Follow the instructions.

knowing how to label images is a tricky decision in itself - experience showed that the detections are better when the labels are applied close-cropped to the cat's head for `Cat-alone` like this:
![image](https://github.com/Charry2014/ai-catflap/assets/58067238/c60d5423-ce57-43c7-9a18-78fa0c2e954a)

For the `Cat-with-mouse`case the labels are centered around the mouse body, but with the cat's eyes in the area
![image](https://github.com/Charry2014/ai-catflap/assets/58067238/f062568c-6abb-453a-84dd-6b1d0d317cbe)

From these images you will also see that one is taken in natural light, and the other is taken with IR illumination from the Pi camera. You will need sample images of all labels in both lighting conditions. 

Perhaps an expert in how these models work can clarify or elaborate what exactly would work best here, this is complex and a field of study in its own right. For now this is OK.

## Export the Images and Labels

Once you have gathered a few images (the more the better) they need to be exported from Label Studio in the Pascal VOC format, which exports the image and a .xml file that describes the labels and areas. Unfortunately the format produced by Label Studio does not work directly with Google TensorFlow Lite so it is necessary to run a script to do the conversion.

On the Mac Label Studio will default to downloading a .zip to the `/Users/[username]/Downloads` directory. Unpack this into the same place and the script looks there to find a **folder** named in the Label Studio format, which is something like this `project-1-at-2023-09-08-08-26-e3408fe3`.  

After unpacking the .zip, simply run the script.
```
project-root# ./bin/buildtflite.sh 
```

The scripting here is very simple and certainly will not work in many situations simply as-is but the important stuff happens in `src/buildtflite/buildtflite.py` and the associated `requirements.txt` to install the dependencies. The attentive amongst you may notice that the versions specified are far from the newest but at least on macOS this is the newest combination that will work.

## Train the Model

The `./bin/buildtflite.sh` script will create the model for you from the labelled image data, however it is worth drawing attention to the iterative nature of this. Take some images, label them, run them in your target system, have that record the images it classifies, take any images that are incorrectly classified by the model and use them as new training images, repeat.

# Live Deployment

Once you have trained a model and have the Pi in place it is time to start using the output to enable and disable the cat flap.

## Modifying the Cat Flap
-- work in progress --

[Sure Flap cat flap disassembly instructions on the FCC website](https://fccid.io/XO9-FLAP-1001/Internal-Photos/INTERNAL-PHOTOS-1238385). If you have never used the FCC website it is sometimes very helpful for finding out the internal details of many products.


