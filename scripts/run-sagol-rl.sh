#!/bin/bash

if [[ -f /opt/ros/foxy/setup.bash ]]; then
ROS_DISTRO=foxy
elif [[ -f /opt/ros/galactic/setup.bash ]]; then
ROS_DISTRO=galactic
fi
source /opt/ros/$ROS_DISTRO/setup.bash
source /sagol_ws/install/setup.bash

cd /sagol_ws/src/sagol
python3 sagol/rl_tf_driver.py --model savedmodel/models/ $@
