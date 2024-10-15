#!/bin/bash

if [[ -f /opt/ros/foxy/setup.bash ]]; then
ROS_DISTRO=foxy
elif [[ -f /opt/ros/galactic/setup.bash ]]; then
ROS_DISTRO=galactic
fi
source /opt/ros/$ROS_DISTRO/setup.bash
source /sagol_ws/install/setup.bash

cd /sagol_ws
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
#ros2 run sagol wall_follow
cd src/sagol/sagol
python3 sb3_driver.py --load saved_models/best.zip --no-safety eval $@
