#!/bin/bash

if [[ -f /opt/ros/foxy/setup.bash ]]; then
ROS_DISTRO=foxy
elif [[ -f /opt/ros/galactic/setup.bash ]]; then
ROS_DISTRO=galactic
fi
source /opt/ros/$ROS_DISTRO/setup.bash
#source /f1tenth_ws/install/setup.bash
source /sim_ws/install/setup.bash
#source /autoware/install/setup.bash

cd /sim_ws
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
