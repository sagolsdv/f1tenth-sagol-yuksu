#!/bin/bash

if [[ -f /opt/ros/foxy/setup.bash ]]; then
ROS_DISTRO=foxy
elif [[ -f /opt/ros/galactic/setup.bash ]]; then
ROS_DISTRO=galactic
fi
source /opt/ros/$ROS_DISTRO/setup.bash
source /sagol_ws/install/setup.bash

cd /sagol_ws
ros2 run sagol wall_follow
