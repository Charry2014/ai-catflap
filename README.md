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

As good as the Pi OS is, it is still more comfortable to develop on a proper desktop system - this project is developed on either a Debian Linux (Ubuntu) desktop or macOS for this. Here comes the first gotcha - this is now **cross-platform development**. Not every version of Python is easily installable on the Pi, and not every package is built for it either, so for a long time the rule was to use exactly the version available on the Pi - Python 3.9.10 - and hope. That is no longer necessary. The whole thing now runs in a Docker container, the same image on the desktop and on the Pi, so the versions can no longer drift apart. See [Running in a Container](#running-in-a-container) below. The only part that still wants its own old environment is training the model.

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

Training is the one part that stays on the old software: `tflite-model-maker` wants an old TensorFlow and an old Python, which is why `src/buildtflite/` has an environment of its own and nothing to do with the one that runs the cat flap. That is fine, because all it has to do is hand over a `.tflite` file and the running system reads that file, whichever version made it. Nothing about the model or the labels changed when the detector moved to `ai-edge-litert`.

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

There are two ways to get this running. The first, and the one to use, is the Docker container. The second is the way it was done before the container existed, by installing everything by hand into a virtual environment, and it is described afterwards for completeness - and because it is a good record of what the container is doing for you.

### Running in a Container

The same image runs on the desktop (development, tests) and on the Pi (the real thing), so the Python and package versions are the same in both places. Every package is pinned in `docker/requirements.txt` - regenerate it with the command at the top of `docker/requirements.in` when you want to move forward - and the image uses Python 3.12. One gotcha: the Pi needs a **64-bit** OS, there are no packages for the 32-bit one.

```
ai-catflap# docker compose run --rm --build test             # run the tests, on the desktop or on the Pi
ai-catflap# docker compose up --build                        # run against a camera stream, with a simulated cat flap (desktop)
ai-catflap# STREAM=/app/recordings/clip.mp4 docker compose up   # replay a recorded video (put it in ./recordings first)
```

Settings are environment variables, or a `.env` file next to `docker-compose.yml` - copy `.env.example` to get started. The model, the three JSON files, `recordings/` and `log/` are all mounted from your checkout, so a new model or a new threshold needs a restart but not a rebuild.

On the Pi the camera stays outside the container. `picamera2` needs the libcamera bindings from the Pi OS, which will not install into the container's Python (looking at you again, `libcamera`). Instead `docker/mjpeg_server.py` runs on the Pi itself and serves the camera as a web stream at `http://[your.pi.ip.address]:8000/stream.mjpg`, and the container reads that. There is a systemd unit, `docker/catflap-camera.service`, to start it at boot. If the stream drops the program reconnects rather than giving up. Then start the cat flap with -

```
ai-catflap# docker compose -f docker-compose.yml -f docker-compose.pi.yml up -d --build
```

The `docker-compose.pi.yml` part hands the container the GPIO (`/dev/gpiomem`) and sets `CATFLAP_GPIO=hardware`, which means it refuses to start at all if it cannot control the relay, rather than quietly running with a cat flap that never locks. Without it, `CATFLAP_GPIO=auto`, the relay is simulated, which is just what you want on the desktop. The `restart: unless-stopped` setting replaces the `until` loop in `bin/catflap.sh`, and a health check marks the container as unhealthy if the log goes quiet, which means a stuck camera or a stuck loop. To build the image for the Pi on a desktop use `docker buildx build --platform linux/arm64 -t ai-catflap .`.

A word of honesty: the Pi camera server, the GPIO from inside the container and the arm64 build have not yet been tried on a real Pi. The tests cover everything else, but a relay is not something to take on trust, so check by hand that the flap really locks before relying on it. A Pi 5 will also need `GPIO_LIB=rpi-lgpio` and the `/dev/gpiochip*` device instead, see `docker-compose.pi.yml`.

### Running Without a Container

If you would rather not use the container, then having pulled the sources from GitHub start installing the required packages. This is a real pain. It breaks all the time, with more or less every update of any package something stops working. You will notice in `src/catflap/requirements.txt` that *everything* is versioned precisely, for exactly that reason. The installation seems even more problematic in a Python virtual environment, but these should be considered an essential for any meaningful development project (see the discussion about Label Studio above) and so it is worth getting this working properly.

The installation script `bin/setup.sh` should install most things you need, and perform a couple of necessary `cp` to finally get everything where it is needed - at least it forms a decent documentation of what needs to be done to get this running. As mentioned above some packages don't install correctly into a virtual environment. Looking at you `libcamera` and `pykms`, this is the most pragmatic 'fix' at this point.

Once everything is installed the script can be started with the following commands - assuming the camera is at index 0.

```
ai-catflap# python3 -m venv venv 
ai-catflap# source venv/bin/activate
ai-catflap# ./bin/catflap.sh -s 0
```
There are many things that can go wrong, but creating the `log` and `recording` directories should be easy to fix.

## Monitoring
As with all installed control devices some monitoring will be needed to detect faults. This is a manual effort, with some automation possible.

1. Logging - all events from the control code are logged. These logs are quite noisy but detailed and could be uploaded to a logging service such as Grafana for better visibility.
2. Recordings - images, and if you ask for them short videos, are recorded of the cats comings and goings, with their classifications from the Tensor Flow Lite model. See discussion below.
3. Message Sequence Diagrams generated from the logs - these are not classic message sequence diagrams but the [PlantUML](https://plantuml.com/sequence-diagram) engine is (mis)used to draw an informative diagram of what happens in a log. See `puml/main.py` for more.
4. Status Webpage - See below

### Overview Website
The Pi can host a small web server that gives an overview of what is happening on the cat flap. This is not secure and intended for use only in your LAN, for example in a [Home Assistant](https://www.home-assistant.io/) dashboard. The webpage shows the camera stream, the tail of the logfile, and the last evaluation result and runs in a Docker container.

When the Docker container running the server is active the site will be available on [http://[your.pi.ip.address]:5000](http://[your.pi.ip.address]:5000)

Here is a short Docker 1.01 helper to get the site started. If you didn't install Docker on your Pi (Bullseye) yet -
```
sudo apt update && sudo apt install -y docker.io docker-compose
```

Then to build the Docker image and run the server -
```
ai-catflap # cd src/website
ai-catflap/src/website # sudo docker build -t webserver .
ai-catflap/src/website # sudo docker run -d -it -p 5000:5000 --rm --name web --mount type=bind,source=[/full/path/to/log/dir]/log,target=/log,readonly webserver
```
Note - remember to change `[/full/path/to/log/dir]` to the full path to your logs directory. This will be something like `/home/pi/projects/catflap2/ai-catflap/log`.

Demystify the Docker command - What this does is run the container as a daemon in the background `-d`; create a STDIN terminal to access the container useful for debugging `-it`; exposing port 5000 in the container as port 5000 on the host `-p 5000:5000`; automatically deleting the container if it stops `--rm`; giving the running instance the name `--name web`; mount the local logs directory into the `/log` location in the container `--mount type=bind,source=[/full/path/to/log/dir]/log,target=/log,readonly`.

Some other useful Docker commands -
* Stop the web server - `sudo docker stop web`
* For debugging get the logs from the container - `sudo docker logs web`. Note that you must remove the `--rm` command line option from the `docker run` command to prevent the removal of the container on stop.
* Clean up/delete a container that is stopped `sudo docker container rm web`.



### Recorded Images
In its normal operation the Python scripting will record all images that are classified into the `recording` directory. These can then be reviewed for false detections and a new model trained. This can be done by mapping the directory onto an NFS share visible over the LAN or by using VNC to connect to the Pi and view directly in the host, or likely many other ways. Either way, these recorded images are a good source of new training data to improve the model.

If a still image is not enough then `--record_video` records a short video of every event. The nice part is that the video starts a few seconds *before* the trigger (`--preroll`, 3 seconds unless you say otherwise), so you see the cat arriving and not just the cat already in the doorway, and it carries on until a few seconds after the last activity (`--postroll`, 5 seconds). With `--record_overlays` the boxes and labels from the model are drawn onto both the images and the video, on the frame the model really looked at. That makes it easy to see why it decided what it decided.

# How Does It Work

The Python script used to be a single thread run loop that got a picture from the camera, detected motion, ran the image through the model and wrote the files, one thing after the other. Simple, but sufficient it was not. Anything slow - the model, writing a JPEG, sending the preview to the website - made the camera wait, so frames were dropped and the recordings had holes in them in exactly the places where the cat was doing something interesting. The cure was to stop doing everything in order and let each job get on with its own thing, handing over what it has made through a small buffer. A slow job then skips frames, rather than stopping the camera.

| Thread | What it does | How it hands over |
|---|---|---|
| capture (`capturethread.py`) | The only code that ever talks to the camera. Takes a picture 15 times a second and nothing else | A ring buffer of the last few seconds of timestamped pictures, the `FrameBus` (`framebus.py`) |
| analysis (`analysisthread.py`) | Takes the newest picture only. While the flap is idle it looks for movement (`motion.py`), once something has moved it runs the model on every picture | Events in a queue |
| main | The state machine, the GPIO relay and any windows. Nothing slow is allowed to run here | - |
| video recorder (`videorecorder.py`) | Reads the pictures straight out of the ring buffer, so it gets every one, and writes the video including the pre-roll | The state machine says `trigger` |
| image writer, website | Writing JPEGs and sending the preview picture, each in a little worker (`backgroundworker.py`) with a small queue | Queues |

If the model is slower than the camera it simply skips pictures. It never queues them up, because an answer about where the cat was a second ago is no use to anybody. Every result is remembered against the time its picture was taken (`annotations.py`), which is how the overlays end up on the right frame. The five minute timeout timer used to change the state machine from a thread of its own, which was a little brave. Now it just posts an event, the same as everybody else, and the state machine handles it on the main thread. The model is loaded once at startup, not every time the state changes. And since we are fond of logs, every 30 seconds there is a line saying how many pictures were captured, analysed, skipped and dropped, and how late the latest result was. If that latency creeps up you know the Pi is struggling.

## The Model and the Detector

Detection is done by `ai-edge-litert`, which is what the TensorFlow Lite runtime has been renamed to and which Google do maintain. The previous version used the `tflite-support` Task library, which stops at Python 3.9 and was the reason the whole project was stuck on that version. The old code is still there (`modules/tflite_detect.py`) and can be chosen with `--detector tflite-support`, but `--detector auto`, the default, uses litert whenever it is installed.

The models and the training did not change at all. `src/buildtflite/` still trains with `tflite-model-maker` in its own old environment and still exports a `.tflite` file, and that file runs unchanged. What the Task library did was nothing magic: resize the picture, read the four outputs of the model (boxes, classes, scores and a count - the model does its own non-max suppression), and look up the class names. All of that is now in `modules/litert_detect.py`, and the class names come from the label list stored inside the model, falling back to `labels.json` if a model has none.

There is one gotcha. The two libraries resize pictures slightly differently, so the scores differ by a few hundredths and a detection that sits right on a threshold, say 0.30, can tip over to 0.32. On a set of 30 made up test pictures the best label was the same 28 times and the median score differed by 0.01 - but those were made up. **Before you trust it with the cat flap, run `src/tools/compare_detectors.py` over the pictures in your `recordings/` directory.** It runs both detectors on every picture and lists the ones where the answer changed. It needs both libraries, so use a Python 3.9 environment: `pip install tflite-support==0.4.3 ai-edge-litert "numpy<2" opencv-python-headless`. If a lot of pictures flip around the numbers in `trigger_config.json` or `eval_config.json` then those want a small adjustment. Nobody has done this comparison with real recordings yet.

## Command Line Arguments

All options default to sensible values so the script can, in most cases, be run with no arguments. In the container you do not type these, they are set from environment variables by `docker/entrypoint.sh`, see `.env.example`.

* **--stream** - Define an input stream which can be a single JPEG image, a recorded video (MP4), a Pi camera stream, or a web stream (`http://`, `rtsp://` or `tcp://`). See the discussion in `imgsrcfactory.py` below.
* **--trigger** - Sets the rectangle that defines the trigger area in the video for the motion detection. The trigger area coordinates are in the form x,y,w,h.
* **--model** - Set a path to the .tflite model file
* **--detector** - `auto`, `litert` or `tflite-support`, see above
* **--record_path** - The directory where images and videos are saved
* **--record_video** - Record a video of every event, starting `--preroll` seconds (default 3) before the trigger and ending `--postroll` seconds (default 5) after the last activity
* **--record_overlays** - Debugging - records images and video with the Tensor Flow results overlayed on them so it can be seen what the model decided was in the image
* **--web** - The address of the monitoring website, for the preview picture
* **--show_trigger** - Debugging - shows the motion detection trigger area in the video stream window
* **--headless** - Debugging - display no windows, so the model can be run headless (with no display)

One more setting is not a command line argument but an environment variable, `CATFLAP_GPIO`. It is `auto` (use the GPIO if there is one, simulate the relay if not), `hardware` (refuse to run without it) or `sim` (never touch the hardware).

## Source File Overview

### main.py 
It all begins here. It opens the image source, loads the model, starts the capture and analysis threads and then sits in the main loop taking events off the queue and handing them to the state machine. If anything goes wrong it still goes through the shutdown steps one at a time, with unlocking the cat flap first, because a cat flap stuck shut is the one failure that is really noticed.

### imgsrcfactory.py
This is a fairly standard factory class that creates an instance of a class based on the abstract base class `class AbstractImageSource(ABC)`. All these classes are image sources, either a single image, a video or a stream. See the various `imgsrc*.py` files for details. The choice of which class will be the image source is crudely but effectively taken based on the stream name given. A number is a camera - the Pi camera if `picamera2` is installed, a webcam if not, which is the case on a desktop and in the container. A name starting `http://`, `rtsp://` and so on is a network stream (`imgsrcstream.py`), and a video file is replayed at the speed it was recorded at so it behaves like a live camera.

### framebus.py, capturethread.py
The camera thread and the buffer it fills. Every picture becomes a `Frame` with an id and the time it was captured, and is made read-only so nobody can scribble on a picture somebody else is still using. Anything that wants to draw on a picture makes a copy.

### analysisthread.py, motion.py
The analysis thread decides what each picture means. While the flap is idle `motion.py` compares the trigger area with a picture from about a second earlier - no gap and slow movement is not seen at all. Once the state machine has left idle the model is run on every picture instead, and the results are sent to the state machine as events.

### states.py
Runs the state machine that locks and unlocks the cat flap. When no motion is detected in front of the camera the state machine is in `Idle` state and the cat flap is unlocked, allowing cats to exit normally. As soon as motion is detected the state machine starts an unlock timer (see `statetimer.py`) and transitions to `MotionLocked` state. Here images are fed through the Tensor Flow Lite model until an evaluation can be made with sufficient confidence (see `evaluation.py`) if there is a cat alone or cat with mouse. 

Once sufficient images have been collected and a suitable confidence has been reached the state machine will transition to either `CatAlone` or `CatWithMouse` state. If the evaluation lands with a cat alone decision then the cat flap is unlocked and kept unlocked until the timeout triggers and returns to `Idle`. If a mouse is seen, then the flap remains locked until the timeout and, you guessed it, a return to `Idle` state.

The state machine only ever sees events, and an event also says whether it was made for the state it finds itself in. If the state has changed in the meantime - the picture was analysed for movement but the machine has since moved on to looking for cats - the result is stale and quietly dropped. The state machine also tells the analysis thread when to start and stop running the model, so the model is not burning the Pi's CPU while nothing is moving.

Simple but effective.

### evaluation.py
Maintains a list of Stats classes for each possible label - `Cat-alone` and `Cat-with-mouse` and evaluates each for a level of certainty. Again, very simple.

### videorecorder.py, image_recorder.py, annotations.py
Everything to do with keeping a record. The `ClipRecorder` reads pictures straight from the ring buffer, so a slow model cannot cause holes in the video, and starts the video before the trigger. A video keeps running for as long as there is something going on and ends a few seconds after. The `ImageRecorder` queues stills and writes them from a thread of its own, naming the file when the picture is taken and not when it gets written. `annotations.py` is the memory of what the model said about which picture, so overlays can be drawn onto the right one.

### backgroundworker.py
A thread with a small queue, for the jobs that are slow but not important. When the queue is full it drops a picture rather than make anyone wait, and counts it, so you see it in the log.

### catflapcontrol.py
The relay. Uses the Pi's GPIO if it can, otherwise pretends - see `CATFLAP_GPIO` above.

### detections.py
Convenience classes that mimic classes used by Google but in a minimalist style, and the function that draws the boxes onto a picture.

### modules/litert_detect.py, modules/tflite_detect.py, modules/detector.py
The two ways of running the model, and the little factory that picks one. `tflite_detect.py` is the Google example stripped down to the absolute minimum. `litert_detect.py` does the same job without the Task library, see "The Model and the Detector" above.

### modules/base_logger.py
A minimal hacky logging wrapper to log both to the console and a file. Generally speaking, logs are awesome in a project like this, and tools like PlantUML (discussed above) and [Grafana](https://grafana.com/) are just magic for making logs actually mean something.

### tools/compare_detectors.py
Runs both detectors over a directory of pictures and shows where they disagree. See "The Model and the Detector" above.

### docker/
Everything the container needs: the pinned `requirements.txt`, the `entrypoint.sh` that turns environment variables into command line arguments, and the camera stream server and service file for the Pi.

# Tests

There are tests, and they need no camera, no Pi and no model training. The quickest way to run them is in the container -

```
ai-catflap# docker compose run --rm --build test
```

or with `pip install pytest` and `pytest test/` in an environment with the packages from `docker/requirements.txt`. One thing to know if you do it by hand is that `python-statemachine` must be exactly 2.0.0. Version 2.1 and later give an error as soon as the state machine starts, the state classes clash with what the library now expects. The old `requirements.txt` asked for 2.1.2, which cannot have been working. The tests cover the frame buffer, the threads, the state machine driven by events, the video recorder and its pre-roll, the web camera stream, the detector against the real model, and one run of the whole thing from a video file to recorded images and a video - with a faked model, that part, so it checks the plumbing and not the cat.
