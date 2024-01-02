# ano(The)r AI Catflap
An AI based mouse detecting cat flap extension for the Sure Flap cat flap. Inspired by a [project by Gerrit Bojen](https://towardsdatascience.com/keep-your-home-mouse-free-with-an-ai-powered-cat-flap-a67c686ce394). It includes collecting data, training a model, deploying to the Pi, monitoring performance.

Describing this as an AI project is a bit like saying that driving a car is just a matter of turning the wheel. There are many skills to master and despite the plentiful YouTube videos showing you ML on the Pi in 10 minutes, this has been a long project with many problems overcome. The machine learning part of this is by far not the most difficult, thanks to the work of those wonderful people at Google - but getting to that point is hard.

# Introduction

During the COVID pandemic we got two cats as a way of distracting the children from the horrors of being locked up at home together. The cats, being cats, set about discovering the neighbourhood and then filling our house with all manner of furry critters. These poor animals are unfortunate trophies of their hunting ability, to show us how much they love us, or to help us hone our own hunting abilities - depending what you believe about cats' motivations. Anyway, we did not find any of these options particularly appealing - and certainly not practicing our hunting ability in our own living room. 

Our cats are good hunters. We have had many furry critters of all kinds - big and small, small rats, voles, even a baby sand martin. Some have been very much alive - running around the house, eating the cats' food, climbing the curtains, making nests under the sofa - and we have indeed honed our ability to catch and eject these. We are first aware of the presence of a furry guest when we see the cats demonstrating hunting behaviours around the house, then we step in. Worse, however, are the ones alive enough to escape the cat, but then die under the sofa - you can imagine what happens next.

Given this situation we had to do something. Being a pragmatist the first step was to restrict the cat flap to exit-only, but being an engineer this became a challenge.

# Getting Started
## Background
There are several projects on the internet that describe getting AI (object detection, to be more precise) to work on the Raspberry Pi. This one is a little different as it describes how to get a product working, using a custom model, installed in situ, doing something useful. There are many stages to this.

Here, in approximate sequence of necessity, are those stages. As mentioned above, these instructions attempt to describe all the steps necessary to get this project working, including training the model, through to installation and monitoring. If you need more detail, just ask.

## Setup a Development Environment

As good as the Pi OS is, it is still more comfortable to develop on a proper desktop system - this project is developed on either a Debian Linux (Ubuntu) desktop or macOS for this. Here comes the first gotcha - this is now **cross-platform development**. not every version of Python is easily installable on the Pi, so you will want to use the same version on the desktop as available on the Pi - currently Python 3.9.10.

Then for an IDE - projects for [Visual Studio Code](https://code.visualstudio.com/) are included, and the [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) plugin is very comfortable for debugging on the Pi. From the command palette you can then Add Remote host and enter `<pi_username>@<your.pi.ipaddress>`. The Pi have SSH keys generated with `ssh-keygen` and then used for enabling [key based SSH login](https://www.geekyhacker.com/2021/02/15/configure-ssh-key-based-authentication-on-raspberry-pi/) to the Pi and for GitHub.

Oh, and of course, only develop Python inside a virtual environment.

## Data Collection

Without data there is no machine learning, step 1 is get recordings of the cats approaching the camera. Set up a Pi with a camera and add some motion detection software to it. This can be cobbled together using the Raspberry Pi example for streaming from a camera, and the OpenCV example for motion detection. The stream was very useful to be able to monitor the camera live from anywhere on the LAN, very handy for early understanding how the camera siting was working, or not. Recording short, 10 second, videos of the cats approaching (and other sources of motion) is very useful for understaning the approach paths of the cats and sources of false triggers.

Credits here go to the [Pi standard streaming example](http://picamera.readthedocs.io/en/latest/recipes2.html#web-streaming) - detailed instructions on [Random Nerd Tutorials](https://randomnerdtutorials.com/video-streaming-with-raspberry-pi-camera/). Note - with the switch to Bullseye the operation of the Pi camera changed completely and these instructions may not work any more.

On to the stream was a client that detected motion and recorded a 10 second video every time motion was detected. From here the first set of images were collected.

You will want to make sure the Pi is mounted in the right place so the camera can see the cats, they need to be approaching the camera for a few seconds so you get some good clean shots of the cat's face, well exposed, and of course you want to make sure the Pi is protected from the weather, and has a wired LAN connection.

From the videos a series of still images was captured using VLC

* Install [VLC](https://www.videolan.org/vlc/) - follow the instructions on the site

Start up VLC and configure to export JPEG files jpg into the folders above. TensorFlow Lite does not accept PNG so JPG is important! Go to VLC -> `Preferences` -> `Video tab` ->  and set the folder to somewhere sensible, the format to `JPEG`, the prefix to `$N-` so the original video name is retained in the snapshot filename and tick the `sequential numbering` box. Run the video and make a series of separate images from the video - every time the cat's aspect or position has changed a little, you can take another separate image from the video. This helps quickly build up a bigger data set.

Note that Tensor Flow Lite works only with JPEG images, not PNG.

## Image Labelling - Preparation

The following labels are used - `Cat-alone`, `Cat-with-mouse`, `Cat-body`, `Background` categories. Collect as many images as your patience allows in all categories and add them to the labelling tool.

The labels explained -
* `Cat-alone` - The cat's mouth is clearly visible and there is nothing in it
* `Cat-with-mouse` - The cat's mouth is clearly visible and there is a mouse, or some other unfortunate creature, there
* `Cat-body` - Some other part of the cat is visible, not the mouse
* `Background` - No part of a cat is visible, this is a false trigger and ignored in subsequent processing

There are probably other ways of labelling things - but `Cat-alone` and `Cat-with-mouse` are the essentials. A third label of `Nothing-to-see-here` could have been sufficient to handle all other cases, ie. it may not have been necessary to separate into `Cat-body` as well. If you have the motivation you can play around with this.

## Image Labelling - Label Studio

This is a very important step, where the Tensor Flow Lite model 'learns' to distinguish one image from another. By taking enough images showing the full variability of the features that are interesting to detect the model can then determine for itself what is in front of the camera. Label Studio is a web based tool that runs on the local machine. Images can be imported and labelled - labelling involves drawing a rectangle around the important features, and telling Label Studio which of the four labels is there.

* Install [Label Studio](https://labelstud.io/) in another Python virtual environment, not the one used to run the catflap script. This is because updating Label Studio, which makes sense to do, may update packages used by the cat flap and thus most likely break it. See the discussion below about installing the packages for the cat flap scripts.

Command lines using `pip` given below.
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

For the `Cat-with-mouse`case the labels are centered around the mouse body, but with the cat's eyes in the area. 

![image](https://github.com/Charry2014/ai-catflap/assets/58067238/f062568c-6abb-453a-84dd-6b1d0d317cbe)

Perhaps it would also be better to include the entire cat's head as in the `Cat-alone` example above, but this would be something for experimentation. Generally, all these models like `efficientdet_lite0` are pre-trained on vast data sets and to know for sure which is better in any given circumstance is hard. One universal truth remains - more data is always better. The sample images must cover as many environmental and edge cases as possible - natural light, IR illumination from the Pi camera, each cat you have, sunlight, shade and whatever else can happen. A colleague once told a story of their cat coming home with a snake. You get the idea. You will need sample images of all labels in all conditions. 

Perhaps an expert in how these models work can clarify or elaborate what exactly would work best here, this is complex and a field of study in its own right. For now this is OK.

## Export the Images and Labels

Once you have gathered a few images (the more the better) they need to be exported from Label Studio in the `Pascal VOC` format, which exports the image and a .xml file that describes the labels and areas. Unfortunately the format produced by Label Studio does not work directly with Google TensorFlow Lite so it is necessary to run a script to do the conversion.

On the Mac Label Studio will default to downloading a .zip to the `/Users/[username]/Downloads` directory. Unpack this into the same place and the script looks there to find a **folder** named in the Label Studio format, which is something like this `project-1-at-2023-09-08-08-26-e3408fe3`.  

After unpacking the .zip, simply run the script.
```
project-root# ./bin/buildtflite.sh 
```

The scripting here is very simple and certainly will not work in many situations simply as-is but the important stuff happens in `src/buildtflite/buildtflite.py` and the associated `requirements.txt` to install the dependencies. The attentive amongst you may notice that the versions specified are far from the newest but at least on macOS this is the newest combination that will work.

## Train the Model
The AI in this project is based on [Google's Tensor Flow Lite project](https://www.tensorflow.org/lite) and the smallest `efficientdet_lite0` model. See `src/buildtflite/build_tflite.py` for more details. As mentioned, it is an object detection AI, which for me was a fairly random choice as it just sounded more plausible than the other choice of image classification. The training code is based on the [Google Colab example](https://colab.research.google.com/github/khanhlvg/tflite_raspberry_pi/blob/main/object_detection/Train_custom_model_tutorial.ipynb#scrollTo=Jbl8z9_wBPlr) for detecting the Android figurine. The online Colab example for training a model seemed to break a while back, maybe it got fixed again. While it worked it was really useful - using Google's GPU power to train a model is faster than using your local machine, quite likely, and leaves your computer free for other things.

The `./bin/buildtflite.sh` script will create the model for you from the labelled image data, however it is worth drawing attention to the iterative nature of this. Take some images, label them, run them in your target system, have that record the images it classifies, take any images that are incorrectly classified by the model and use them as new training images, repeat.

# Live Deployment

Once you have trained a model and have the Pi in place it is time to start using the output to enable and disable the cat flap.

## Modifying the Cat Flap
The Sure Flap cat flap is a great product, and here is a shout out to the company as they are awesome because they even ship spare parts allowing users to repair their products which is fantastic. Nevertheless it must be stated clearly that **following these instructions will obviously void any warranty on your cat flap and you do so at your own risk**. The modification is simple though and should be reversible - we need to just take control of the cat flap lock and unlock mechanism at the bottom of the cat door.

You can see the [Sure Flap cat flap disassembly instructions on the FCC website](https://fccid.io/XO9-FLAP-1001/Internal-Photos/INTERNAL-PHOTOS-1238385). If you have never used the FCC website it is sometimes very helpful for finding out the internal details of many products. We will cut one of the wires and feed it through the normally closed terminals on the relay (COM1 and NC1 terminals in the image below), using some wire to extend the cable.

The modifications can be made with the cat flap in place, just undo the four screws holding the front plate on. Two at the top are inside the battery compartment, two at the bottom are behind the pop-out covers.
![20231217_172103](https://github.com/Charry2014/ai-catflap/assets/58067238/cf14d0df-3d7c-4fa4-969b-2cfb2d589ee7)

The relay is a [Debo 2 channel](https://www.reichelt.de/entwicklerboards-relais-modul-2-channel-5-v-srd-05vdc-sl-c-debo-relais-2ch-p242810.html?&nbc=1) that is connected to the Pi's 5V and GND lines, and of course the GPIO. 
![image](https://github.com/Charry2014/ai-catflap/assets/58067238/c7b4b85e-a5fe-4dda-8ff4-da00cbc959ff)

As the Pi and the relay are on the outside of the house it is necessary to drill a hole through the flap to pass the wires from the flap control motor. That's it.

The choice of wires is arbitrary but choosing the red wire has the advantage that it may be possible to monitor the voltage on the wire to see if the flap is trying to open the locking mechanism. As you can see in the photos below I cut the black wire.

## Installation
In the photo below you can see the installation. Here a small piece of circuit board is used to hold a socket to short the wire together again if ever it should be necessary to disconnect the Pi. 
![20231217_172302](https://github.com/Charry2014/ai-catflap/assets/58067238/9176da86-8b6b-4c17-8cfa-fed9109dbaa7)
There is plenty of space inside the housing for the extra wires, and the hole drilled for the wires to the relay is in the bottom left. The wires are held in place with a piece of duct tape. Very elegant.

## Running the AI

Having pulled the sources from GitHub start installing the required packages. This is a real pain. It breaks all the time, with more or less every update of any package something stops working. You will notice in `src/catflap/requirements.txt` that *everything* is versioned precisely, for exactly that reason. The installation seems even more problematic in a Python virtual environment, but these should be considered an essential for any meaningful development project (see the discussion about Label Studio above) and so it is worth getting this working properly.

The installation script `bin/setup.sh` should install most things you need, and perform a couple of necessary `cp` to finally get everything where it is needed - at least it forms a decent documentation of what needs to be done to get this running. As mentioned above some packages don't install correctly into a virtual environment. Looking at you `libcamera` and `pykms`, this is the most pragmatic 'fix' at this point.

Once everything is installed the script can be started with the following commands - assuming the camera is at index 0.

```
ai-catflap# python3 -m venv venv 
ai-catflap# source venv/bin/activate
ai-catflap# ./bin/mousetest.sh -s 0
```
There are many things that can go wrong, but creating the `log` and `recording` directories should be easy to fix.

## Monitoring
As with all installed control devices some monitoring will be needed to detect faults. This is a manual effort, with some automation possible.

1. Logging - all events from the control code are logged. These logs are quite noisy but detailed and could be uploaded to a logging service such as Grafana for better visibility.
2. Recordings - images are recorded of the cats comings and goings, with their classifications from the Tensor Flow Lite model. See discussion below.
3. Message Sequence Diagrams generated from the logs - these are not classic message sequence diagrams but the [PlantUML](https://plantuml.com/sequence-diagram) engine is (mis)used to draw an informative diagram of what happens in a log. See `puml/main.py` for more.
4. Status Webpage - to-do but coming soon - the Pi hosts a webpage that shows the camera stream, the tail of the logfile, and the last evaluation result. This can then be hosted in, for example, a [Home Assistant](https://www.home-assistant.io/) dashboard, for example.

These are future extensions.

### Recorded Images
In its normal operation the Python scripting will record all images that are classified into the `recording` directory. These can then be reviewed for false detections and a new model trained. This can be done by mapping the directory onto an NFS share visible over the LAN or by using VNC to connect to the Pi and view directly in the host, or likely many other ways. Either way, these recorded images are a good source of new training data to improve the model.

# How Does It Work

The Python script is essentially a single thread run loop that gets a picture from the camera, detects motion (by comparing this new image to the previous one), and then runs the image through the Tensor Flow Lite model. Simple but sufficient. The image classification and probability are fed into a simple evaluation unit that determines is it certain enough that the cat is alone, or is there a too high chance that there is something else hitching a lift. A state machine determines if the cat flap should be locked or unlocked.

## Command Line Arguments

All options default to sensible values so the script can, in most cases, be run with no arguments. 

* **--stream** - Define an input stream which can be a single JPEG image, a recorded video (MP4), or a Pi camera stream. See the discussion in `imgsrcfactory.py` below.
* **--trigger** - Sets the rectangle that defines the trigger area in the video for the motion detection. The trigger area coordinates are in the form x,y,w,h.
* **--model** - Set a path to the .tflite model file
* **--record_overlays** - Debugging - records images with the Tensor Flow results overlayed on them so it can be seen what the model decided was in the image
* **--show_trigger** - Debugging - shows the motion detection trigger area in the video stream window
* **--headless** - Debugging - display no windows, so the model can be run headless (with no display)


## Source File Overview

### main.py 
It all begins here and runs the main loop from here. See the function `motionDetection` for the details. Worthy of note is the extremely hacky `image_classification` function that can be used to re-run the classification on a saved image for testing.

### imgsrcfactory.py
This is a fairly standard factory class that creates an instance of a class based on the abstract base class `class AbstractImageSource(ABC)`. All these classes are image sources, either a single image or a stream. See the various `imgsrc*.py` files for details. The choice of which class will be the image source is crudely but effectively taken based on the stream name given.

### states.py
Runs the state machine that locks and unlocks the cat flap. When no motion is detected in front of the camera the state machine is in `Idle` state and the cat flap is unlocked, allowing cats to exit normally. As soon as motion is detected the state machine starts an unlock timer (see `statetimer.py`) and transitions to `MotionLocked` state. Here images are fed through the Tensor Flow Lite model until an evaluation can be made with sufficient confidence (see `evaluation.py`) if there is a cat alone or cat with mouse. 

Once sufficient images have been collected and a suitable confidence has been reached the state machine will transition to either `CatAlone` or `CatWithMouse` state. If the evaluation lands with a cat alone decision then the cat flap is unlocked and kept unlocked until the timeout triggers and returns to `Idle`. If a mouse is seen, then the flap remains locked until the timeout and, you guessed it, a return to `Idle` state.

Simple but effective.

### evaluation.py
Maintains a list of Stats classes for each possible label - `Cat-alone` and `Cat-with-mouse` and evaluates each for a level of certainty. Again, very simple.

## detections.py
Convenience classes that mimic classes used by Google but in a minimalist style.

## modules/tflite_detect.py
Here is the minimal Tensor Flow Lite detection code. It is the Google example stripped down to the absolute minimum.

## modules/base_logger.py
A minimal hacky logging wrapper to log both to the console and a file. Generally speaking, logs are awesome in a project like this, and tools like PlantUML (discussed above) and [Grafana](https://grafana.com/) are just magic for making logs actually mean something.



