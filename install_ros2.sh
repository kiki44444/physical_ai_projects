#!/bin/bash
set -e

# 1. Check Ubuntu version
source /etc/os-release

if [[ "$ID" != "ubuntu" || "$VERSION_ID" != "22.04" ]]; then
    echo "Error: ROS 2 Humble requires Ubuntu 22.04."
    exit 1
fi

# 2. Install dependencies
apt update
apt install -y locales software-properties-common curl

# 3. Configure locale
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# 4. Enable universe repository
add-apt-repository -y universe

# 5. Add ROS 2 GPG key
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

# 6. Add ROS 2 repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${UBUNTU_CODENAME} main" \
    > /etc/apt/sources.list.d/ros2.list

# 7. Update packages
apt update
apt upgrade -y
# 8. Install ROS 2 Humble
apt install -y ros-humble-desktop

# 9. Install development tools
apt install -y \
    python3-rosdep \
    build-essential

# 10. Initialize rosdep
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    rosdep init
fi

rosdep update

# 11. Source ROS 2
source /opt/ros/humble/setup.bash

# 12. Create ROS 2 workspace
mkdir -p ~/ros2_ws/src

cd ~/ros2_ws/src

# 13. Create robot_control package
if [ ! -d "robot_control" ]; then
    ros2 pkg create robot_control \
        --build-type ament_python \
        --dependencies rclpy sensor_msgs
else
    echo "robot_control package already exists. Skipping."
fi

# 14. Configure bashrc
ROS_SOURCE="source /opt/ros/humble/setup.bash"

grep -qxF "$ROS_SOURCE" ~/.bashrc || echo "$ROS_SOURCE" >> ~/.bashrc

# 15. Verify installation
echo "ROS_DISTRO=$ROS_DISTRO"

ros2 --help

echo "ROS 2 Humble installation completed!"
echo "Workspace: ~/ros2_ws"
echo "Package: robot_control"