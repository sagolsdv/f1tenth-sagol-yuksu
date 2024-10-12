#!/usr/bin/env bash

# give docker permission to use X
sudo xhost +si:localuser:root
if [[ ! -d /sys/fs/cgroup/systemd ]]; then
	sudo mkdir /sys/fs/cgroup/systemd
	sudo mount -t cgroup -o none,name=systemd cgroup /sys/fs/cgroup/systemd
fi

GPU=
if [[ `uname -p` == "x86_64" ]]; then
    GPU="--gpus all"
fi

# run container with privilege mode, host network, display, and mount workspace on host
sudo docker run --rm --name sagoldev $GPU -it --privileged --network host -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix -v /dev:/dev \
    -v $(pwd)/src:/sagol_ws/src \
    sagol:base
