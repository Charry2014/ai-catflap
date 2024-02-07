#!/bin/bash
# To be run from the website directory
LOG_PATH="$PWD"/../../log

sudo docker stop web
sudo docker build -t webserver .
sudo docker run -d -it -p 5000:5000 --rm --name web --mount type=bind,source=$LOG_PATH,target=/log,readonly webserver