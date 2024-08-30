#!/bin/bash

ifconfig eth0 192.168.0.100

source /opt/ros/galactic/setup.bash
source /f1tenth_ws/install/setup.bash
#source /sim_ws/install/setup.bash
#source /autoware/install/setup.bash

cd /f1tenth_ws
ros2 launch f1tenth_stack bringup_launch.py
