#!/usr/bin/env bash

# give docker permission to use X
sudo xhost +si:localuser:root

# run container with privilege mode, host network, display, and mount workspace on host
sudo docker run -it --privileged --network host -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix -v /dev:/dev \
    -v $(pwd)/src:/sagol_ws/src \
    sagol:base
