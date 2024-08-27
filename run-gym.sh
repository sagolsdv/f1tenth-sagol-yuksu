#!/bin/bash

source /opt/ros/galactic/setup.bash
#source /f1tenth_ws/install/setup.bash
source /sim_ws/install/setup.bash
#source /autoware/install/setup.bash

cd /sim_ws
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
