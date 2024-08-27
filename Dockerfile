FROM ubuntu:20.04

RUN apt update && apt-get -y install --no-install-recommends locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
RUN apt-get -y install --no-install-recommends software-properties-common
RUN add-apt-repository universe
RUN apt update && apt-get -y install --no-install-recommends curl
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update

# dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y git \
                       nano \
                       vim \
                       python3-pip \
                       libeigen3-dev \
                       tmux \
                       ros-galactic-desktop ros-galactic-ros-base ros-dev-tools \
                       ros-galactic-rosbridge-server ros-galactic-control-msgs ros-galactic-serial-driver \
                       ros-galactic-tf2-geometry-msgs ros-galactic-ackermann-msgs ros-galactic-joy ros-galactic-nav2-map-server ros-galactic-rviz2
RUN apt-get install -y ros-galactic-urg-node ros-galactic-diagnostic-updater ros-galactic-test-msgs
RUN apt-get install -y python3-rosdep
                       
SHELL ["/bin/bash", "-c"] 
RUN apt-get -y dist-upgrade
RUN pip3 install transforms3d
RUN rosdep init
ENV ROS_DISTRO=galactic

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
    rosdep install --from-paths src -i -y && \
    colcon build

# f1tenth gym
#RUN cd / && git clone https://github.com/f1tenth/f1tenth_gym
#RUN cd /f1tenth_gym && \
#    pip3 install -e .

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


# f1thenth_gym
RUN cd / && git clone https://github.com/f1tenth/f1tenth_gym && \
    source /opt/ros/galactic/setup.bash && \
    cd f1tenth_gym && pip3 install -e .

# f1thenth_gym_ros
RUN cd / && mkdir -p sim_ws/src && \
    cd sim_ws/src && \
    git clone https://github.com/f1tenth/f1tenth_gym_ros && \
    source /opt/ros/galactic/setup.bash && \
    cd .. && rosdep install -y --from-paths src --rosdistro $ROS_DISTRO && \
    colcon build
    
# install slam
RUN pip3 install onnx
RUN apt install ros-galactic-slam-toolbox -y

RUN cd / &&\
    source /opt/ros/galactic/setup.bash && \
    cd f1tenth_gym && pip3 install .

# odometer adjust
COPY odom_adjust.patch /f1tenth_ws/src/f1tenth_system/vesc/odom_adjust.patch
RUN source /opt/ros/galactic/local_setup.bash && \
    cd /f1tenth_ws/src/f1tenth_system/vesc && \
    patch -p1 < odom_adjust.patch && \
    cd /f1tenth_ws && \
    colcon build


COPY config/joy_teleop.yaml /f1tenth_ws/install/f1tenth_stack/share/f1tenth_stack/config/joy_teleop.yaml
COPY run.sh /
COPY run-*.sh /

WORKDIR '/'
ENTRYPOINT ["/bin/bash"]
