#!/bin/bash

source /opt/ros/galactic/setup.bash
source /sagol_ws/install/setup.bash

cd /sagol_ws
ros2 run sagol wall_follow
