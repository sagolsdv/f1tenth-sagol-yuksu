FROM ubuntu:20.04

RUN apt update && apt-get -y install --no-install-recommends locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN apt-get -y install --no-install-recommends software-properties-common
RUN add-apt-repository universe
RUN apt-get -y install --no-install-recommends curl
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update  --fix-missing

                       
ENV ROS_DISTRO=galactic
SHELL ["/bin/bash", "-c"]

# ROS2 dependencies
RUN apt-get install -y git nano vim python3-pip libeigen3-dev tmux
RUN apt-get install -y \
    ros-galactic-desktop ros-galactic-ros-base ros-dev-tools \
    ros-galactic-rosbridge-server ros-galactic-control-msgs ros-galactic-serial-driver \
    ros-galactic-tf2-geometry-msgs ros-galactic-ackermann-msgs ros-galactic-joy ros-galactic-nav2-map-server ros-galactic-rviz2 \
    ros-galactic-urg-node ros-galactic-diagnostic-updater ros-galactic-test-msgs \
    ros-galactic-slam-toolbox \
    python3-rosdep

RUN apt-get -y dist-upgrade
RUN pip3 install transforms3d onnx
RUN rosdep init

# autoware
RUN cd / && git clone -b galactic https://github.com/autowarefoundation/autoware.git && \
    cd /autoware && \
    ./setup-dev-env.sh --no-nvidia --no-cuda-drivers -y
RUN cd /autoware && mkdir src && \
    vcs import src < autoware.repos
RUN cd /autoware && source /opt/ros/galactic/setup.bash && \
    rosdep update --include-eol-distros && \
    rosdep install -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO -r 
RUN pip install setuptools==65.5.0
RUN ulimit -c unlimited
RUN cd /autoware && source /opt/ros/galactic/setup.bash && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN pip install "wheel<0.40.0"

# f1tenth system
RUN mkdir -p /f1tenth_ws/src
RUN cd /f1tenth_ws && colcon build
RUN cd /f1tenth_ws/src && \
    git clone https://github.com/f1tenth/f1tenth_system.git && \
    cd f1tenth_system && \
    git submodule update --init --force --remote
RUN cd /f1tenth_ws && \
    source /opt/ros/galactic/local_setup.bash && \
    rosdep update && \
    rosdep install --from-paths src -i -y

# odometer adjust
COPY odom_adjust.patch /f1tenth_ws/src/f1tenth_system/vesc/odom_adjust.patch
RUN source /opt/ros/galactic/local_setup.bash && \
    cd /f1tenth_ws/src/f1tenth_system/vesc && \
    patch -p1 < odom_adjust.patch && \
    cd /f1tenth_ws && \
    colcon build

# f1thenth_gym
RUN cd / && git clone https://github.com/f1tenth/f1tenth_gym && \
    source /opt/ros/galactic/setup.bash && \
    cd f1tenth_gym && pip3 install .

# f1thenth_gym_ros
RUN apt-get install ros-galactic-nav2-lifecycle-manager ros-galactic-joint-state-publisher
RUN cd / && mkdir -p sim_ws/src && \
    cd sim_ws/src && \
    git clone https://github.com/f1tenth/f1tenth_gym_ros && \
    source /opt/ros/galactic/setup.bash && \
    cd .. && \
	rosdep install -i -y --from-paths src --rosdistro $ROS_DISTRO && \
    colcon build

# sagol
RUN pip install tensorflow blosc
RUN pip install gpiod
RUN pip install gymnasium shimmy
#RUN pip install stable_baselines3[extra]
RUN pip install stable_baselines3
RUN pip install numpy==1.23.1
RUN pip install evdev
RUN apt-get install -y screen

RUN mkdir -p /sagol_ws/src
COPY src /sagol_ws/src
RUN source /opt/ros/galactic/setup.bash && \
    cd /sagol_ws && \
	rosdep install -i -y --from-paths src --rosdistro $ROS_DISTRO && \
    colcon build

RUN mkdir -p /sagol_ws/utils
COPY utils /sagol_ws/utils
COPY utils/keep-hokuyo-ip.py /sagol_ws/utils/
COPY utils/mouse2joy.py /sagol_ws/utils/

# FT-Autonomous
#RUN git clone https://github.com/FT-Autonomous/F1Tenth-RL.git
#RUN cd F1Tenth-RL/ &&  git submodule init &&  git submodule update && \
#    source /opt/ros/galactic/setup.bash && \
#    pip3 install numpy scipy numba Pillow gym pyyaml pyglet shapely wandb pylint && \
#    pip install stable-baselines3 shimmy
#  pip install gymnasium==0.28.1
#  pip install numpy==1.23.5

RUN cd / && git clone https://github.com/f1tenth/f1tenth_racetracks.git

    
COPY config/joy_teleop.yaml /f1tenth_ws/install/f1tenth_stack/share/f1tenth_stack/config/joy_teleop.yaml
COPY scripts/run.sh /
COPY scripts/run-mouse.sh /
COPY scripts/run-*.sh /

WORKDIR '/'
#ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT ["/run-all.sh"]
ENTRYPOINT ./run-all.sh
