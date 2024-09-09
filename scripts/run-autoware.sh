#!/bin/bash

if [[ -f /opt/ros/foxy/setup.bash ]]; then
ROS_DISTRO=foxy
elif [[ -f /opt/ros/galactic/setup.bash ]]; then
ROS_DISTRO=galactic
fi
source /opt/ros/$ROS_DISTRO/setup.bash

source /autoware/install/setup.bash


cd /autoware
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=/f1tenth_ws/install/f1tenth_stack/share/f1tenth_stack/config/f1tenth_online_async.yaml 




