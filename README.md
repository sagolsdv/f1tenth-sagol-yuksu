# F1Tenth Sagol Yuksu (사골 육수)

It was developed by Team **Sagol** (사골) for their debut participation in the F1Tenth competition at [IROS 2024](http://iros2024-abudhabi.org/) in Abu Dhabi. The code name of the car was **Yuksu** (육수). The core concept of **Sagol Yuksu** was to build an F1Tenth vehicle powered by a [Raspberry PI 5](https://www.raspberrypi.com/products/raspberry-pi-5/) as the main computing module, while experimenting with Reinforcement Learning using [Stable Baseline3](https://stable-baselines3.readthedocs.io/en/master/).

![Sagol Car - Yuksu Edition 2024](images/sagolcar.png)

## Disclaimer

*DO NOT TRY THIS AT HOME!!!*

This repository serves as an archive of our work, and the **Sagol Yuksu** project contains numerous known design issues in both hardware and software as an experiment. Therefore, it is not recommended to build a car by directly following this repository. Moreover, the system was developed over roughly two months under intense time pressure, by following the “**impulse-driven development**” approach. As a result, coding conventions, code&documentation quality, and overall software/hardware polish was not considered in ths archive repository.

## HW Stack - Bill of Material

* Computing module - [Raspberry PI 5 8GB](https://besomi.com/dedb0624-sc1112-raspberry-pi-5-8gb.html)
  * M.2 Hat - [Raspberry Pi M.2 HAT+](https://besomi.com/raspberry-pi-m-2-hat-desx0013.html)
  * SSD Storage - [SK hynix 256GB PCIe NVMe 2242 SSD (HFM256GD3HX015N)](https://a.co/d/6MHzHub)
  * Power Hat (12V BEC to 5V5A) - [GeeekPi PD Power Expansion Board for Raspberry Pi 5](https://a.co/d/iul6PiQ)
* Motor Controller [VESC 6 MK4](https://trampaboards.com/vesc-6-mkvi-the-amazing-trampa-vesc-6-mkvi-gives-maximum-power-p-27536.html)
* 2D LiDAR - [Hokuyo UST-20LX](https://www.hokuyo-usa.com/products/lidar-obstacle-detection/ust-20lx)
* Power Distribution Board - [Matek System XCLASS PDB FCHUB-12S V2](https://aliexpress.com/item/1005007010163441.html)
  * 12v BEC out to R-Pi5
  * 12v out to LiDAR
  * 1 of other DC distribution to VESC
* Remote Controller(Deadman's Switch) - [Logitech MX Master 3S](https://www.logitech.com/en-us/shop/p/mx-master-3s)
* Battery - [3S Lipo 5000mAh 11.1v 50C](https://a.co/d/cktotOP)
* (Optional) [Infrared Proximity Sensor IR Analog Distance Sensor Infrared distance sensor](https://a.co/d/bm82ltS) for helping to reset position during physical RL sessions.

## SW Stack

### Build & Run

```bash
% docker build -t sagol:base .
% ./run-container.sh
```

### Installation on target HW

```bash
% echo 'KERNEL=="ttyACM[0-9]*", ACTION=="add", ATTRS{idVendor}=="15d1", MODE="0666", GROUP="dialout", SYMLINK+="sensors/hokuyo"' > /etc/udev/rules.d/99-hokuyo.rules
% echo 'KERNEL=="ttyACM[0-9]*", ACTION=="add", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0666", GROUP="dialout", SYMLINK+="sensors/vesc"' > /etc/udev/rules.d/99-vesc.rules
# exec run-container.sh at boot time by using systemctl or crontab
# But you don't have a model to use for autonomous driving yet in
# /sagol_ws/src/sagol/sagol/saved_models/best.zip
# Please follow the next chapter to build a model by sb_driver.py.
```

### Training model on PC

1. Run container on Terminal 1

This script runs sagol:base with enabling X.org connectivity from the container

```bash
% ./run-container.sh 
```

2. Run bash of container on Terminal 2 and kill all processes

The container supopse to run on the target device. In order to use the container for the training mode, it needs to kill the screen sessions for - f1tenth, hokuyo, mouse and sagol.

```bash
% ./join.py # or sudo docker exec -it sagoldev /bin/bash
% pkill screen
```

3. run f1tenth-gym with iros2024 track on Terminal 2

Now you'll run f1tenth gym environment for ROS. You'll want to use simulated track of IROS2024 competition. Edit sim.yaml and start gym.

```bash
% vi /sim_ws/install/f1tenth_gym_ros/share/f1tenth_gym_ros/config/sim.yaml
# modify map_path 
#    from map_path: '/sim_ws/src/f1tenth_gym_ros/maps/levine'
#    to   map_path: '/sagol_ws/src/maps/iros2024'
# and :wq

% ./run-gym.sh
```

The rviz2 GUI will be shown with the IROS2024 track. Click "2D Pose Estimate" button on toolbar and click on map to reset the ego car position.

4. Run bash of container on Terminal 3 to run sb3_driver.py for training mode

```bash
% ROS_DISTRO=galactic
% source /opt/ros/$ROS_DISTRO/setup.bash
% source /sagol_ws/install/setup.bash
% cd sagol_ws/src/sagol/sagol/
% python3 sb3_driver.py training
```
Save your best model into 'saved_models/best.zip'. This inference model will be used by run-sago.sh in your target HW.
