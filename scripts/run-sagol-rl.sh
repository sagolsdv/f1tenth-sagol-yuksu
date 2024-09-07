#!/bin/bash

source /opt/ros/galactic/setup.bash
source /sagol_ws/install/setup.bash

cd /sagol_ws/src/sagol
python3 sagol/rl_tf_driver.py --model savedmodel/models/
