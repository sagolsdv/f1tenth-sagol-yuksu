FROM autodriveecosystem/autodrive_f1tenth_api:2024-iros-practice

ENV ROS_DISTRO=foxy
SHELL ["/bin/bash", "-c"]

RUN apt update && apt-get -y install --no-install-recommends ros-foxy-ackermann-msgs
RUN pip install blosc inputs
RUN pip install tensorflow[and-cuda]

# sb3
RUN pip install gym
RUN pip install stable-baselines3[extra]


# sagol
RUN mkdir -p /sagol_ws/src
COPY src /sagol_ws/src
RUN source /opt/ros/foxy/setup.bash && \
    cd /sagol_ws && \
	rosdep install -i -y --from-paths src --rosdistro $ROS_DISTRO && \
    colcon build

COPY simracing/run-*.sh /

WORKDIR '/'
ENTRYPOINT ["/bin/bash"]
