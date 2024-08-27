#!/bin/bash

source /opt/ros/galactic/setup.bash
#source /f1tenth_ws/install/setup.bash
#source /sim_ws/install/setup.bash
source /autoware/install/setup.bash


cd /autoware
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=/f1tenth_ws/install/f1tenth_stack/share/f1tenth_stack/config/f1tenth_online_async.yaml 




