# physical_ai_projects
### vscode + SSH
1. install remote SSH extension
2. public key of mine
```
# Windows
Get-Content ~\.ssh\id_ed25519.pub

# mac
cat ~/.ssh/id_ed25519.pub
```
3. code
```bash
apt update
apt install -y openssh-server
mkdir -p /run/sshd
/usr/run.sshd or /usr/sbin/sshd

mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo 'YOUR_PUBLIC_KEY' >> ~/.ssh/authorized_keys
# type your public key
chmod 600 ~/.ssh/authorized_keys
```
4. vscode config
```
remote-SSH config file

Host runpod
    HostName {HOST_IP}
    User root
    Port {PORT}
    IdentityFile ~/.ssh/id_ed25519
    
ssh runpod # through terminal
Remote-SSH -> Connect to Host -> runpod # through vscode
```
### Git SSH
public key
- generate a private/public key on the remote server
```
ssh-keygen -t ed25519 -C "{github_email}"
cat ~/.ssh/id_ed25519.pub
```
- add this public key on github
- ssh -T git@github.com

## ROS2 install
Ubuntu 22.04 Jammy
Isaac Sim 4.0 
ROS 2 Humble + Ubuntu 22.04 based on NVIDIA doc
```
apt update
apt install -y locales software-properties-common curl

locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

add-apt-repository universe

apt update
apt install -y curl

curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  > /etc/apt/sources.list.d/ros2.list

# ROS humble
apt update
apt upgrade -y

apt install -y ros-humble-desktop
apt install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    build-essential

source /opt/ros/humble/setup.bash
ros2 --help
printenv ROS_DISTRO
```

```
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ros2 pkg create robot_control --build-type ament_python --dependencies rclpy sensor_msgs
```
```
# colcon install
apt update
apt install -y python3-colcon-common-extensions
colcon --help
```
ROS workspace
```
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
source install/setup.bash

/opt/ros/humble/setup.bash
        ↓
ROS2 Humble 자체를 사용할 수 있게 함

~/ros2_ws/install/setup.bash
        ↓
네가 만든 robot_control 같은 패키지를
ROS2가 찾을 수 있게 함
```
# project #1: isaac-sim-learning
## cloud GPU: Runpod
Create runpod server
1. Get the runpod template for issac sim 4.0 GUI
   - From: https://github.com/Sa3d-99/runpod_noVNC_isaac_sim
2. Change the template container start command into
```
{
  "entrypoint": ["bash", "-c"],
  "cmd": ["curl -fsSL https://raw.githubusercontent.com/Sa3d-99/runpod_noVNC_isaac_sim/main/bootstrap.sh | bash; sleep infinity"]
}
```
3. HTTP service (wait a bit .. )
4. Access the server with SSH (not "SSH exposed to TCP")
5. If you accidently exit the simulator in the server then, ```./isaac-sim.sh --allow-root```

## Franka pandas
### 1. Franka + ground plane + cube
### 2. stardalone python script (/workspace/main.py)
- vi installation
```bash
apt update
apt install -y vim
```
- execution
```python
/isaac-sim/python.sh /workspace/main.py
```
- Trouble shooting
```bash
# exit the GUI
export DISPLAY=:1
echo $DISPLAY
# result :1
/isaac-sim/python.sh /workspace/main.py
```
### 3. Reading the current joint position
- read_joint.py
```python
franka.get_joint_positions()
```
- terminal output
```python
DOF names:
['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
Joint positions:
[ 0.012      -0.57000005  0.         -2.81        0.          3.037
  0.741       0.00760282  0.00763106]

```
- dof: degree of freedom
- units: joint = radian, finger_joing = m
### 4. Move the joint position (Joint space)
```python
# move the joints
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np

# radian
q = np.array([0.0, -0.5, 0.0, -2.0 , 0.0, 1.5, 0.7, 0.04, 0.04])
action = ArticulationAction(joint_positions=q)
franka.apply_action(action)
```
- output
```python
Target: [ 0.   -0.5   0.   -2.    0.    1.5   0.7   0.04  0.04]
Actual: [-4.3578058e-12 -4.9997920e-01  3.6331871e-10 -2.0000052e+00
 -5.4580335e-07  1.5000017e+00  7.0000052e-01  3.9779279e-02
  3.9779294e-02]
```
### 5. Cartesian space + IK solver
- move_cartesian_01.py
```python
from omni.isaac.franka import KinematicsSolver
import numpy as np

ik_solver = KinematicsSolver(franka)

target = np.array([0.5, 0.4, 0.1])

action, success = ik_solver.compute_inverse_kinematics(target_position=target)

print(f"IK success : {success}")
print(f"IK action: {action}")

if success:
    franka.apply_action(action)
    for _ in range(500):
        world.step(render=False)

    actual_pos, actual_ori = (
        franka.end_effector.get_world_pose()
    )

    print(f"Target:", target)
    print(f"Actual:", actual_pos)

simulation_app.close()
```
- output
```python
IK success : True
IK action: {'joint_positions': [-0.2474887004383594, 0.9693817698293395, 0.8151337390306523, -2.223918613446941, 2.837344281500117, 1.884653395022492, 2.2637032751245556], 'joint_velocities': None, 'joint_efforts': None}
Target: [0.5 0.4 0.1]
Actual: [0.4646386  0.37809917 0.0922472 ]
```
- move_cartesian_02.py
    - compute the target joint every simulation steps based on the current robot state
```python
from omni.isaac.franka.controllers import RMPFlowController
import numpy as np

controller = RMPFlowController(name="franka_rmpflow_controller",
    robot_articulation=franka)

# target gripper coordinate (x, y, z)
target = np.array([0.4, 0.2, 0.5])

for _ in range(500):
    action = controller.forward(target_end_effector_position=target)
    franka.apply_action(action)
    world.step(render=False)

print(f"joint pos: {franka.get_joint_positions()}")
print(f"gripper pos: {franka.end_effector.get_world_pose()[0]}")

simulation_app.close()
```
- output
```bash
Joint position: [ 0.16466147 -0.7054785   0.28579456 -2.478773    0.05316455  2.3896084
  0.7499266   0.00760282  0.00760279]
Gripper position: [0.37350965 0.20253271 0.53297734]
```

### 6. Pick-and-place
- 05_pick_and_place.py
- output
```
pre grap
Target: [0.5   0.    0.225]
Actual: [0.49524176 0.00759575 0.2664331 ]
descend
Target: [0.5   0.    0.075]
Actual: [0.5040984  0.00759774 0.11650084]
close gripper
lift
Target: [0.5   0.    0.325]
Actual: [ 4.8856360e-01 -9.2720074e-06  3.6509696e-01]
Cube position after lift [4.9999970e-01 4.2551530e-08 2.4999926e-02]
 move to target
Target: [0.4   0.3   0.325]
Actual: [0.38934413 0.29515666 0.3650151 ]
lower
Target: [0.4   0.3   0.075]
Actual: [0.40588713 0.2989509  0.11627045]
open gripper
retreat
Target: [0.4   0.3   0.325]
Actual: [0.36543915 0.32712418 0.36237168]
final cube position: [4.9999970e-01 4.2551530e-08 2.4999926e-02]
```
## ROS2
### 1. publish
- 06_ros2_publish.py
- How to execute
  ```bash
  source /opt/ros/humble/setup.bash
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export ROS_DOMAIN_ID=0
  /isaac-sim/python.sh 06_ros2.py
  ```
### 2. Subscribe
```
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
ros2 topic list
ros2 topic echo /joint_states
```

### 3. command
- 07_ros2_command.py
- How to execute
- ```
  source /opt/ros/humble/setup.bash
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export ROS_DOMAIN_ID=0
  /isaac-sim/python.sh 06_ros2.py
  ```
- output
```
Current joints: [ 0.012  -0.57    0.     -2.81    0.      3.037   0.741   0.0076  0.0076]
Current joints: [ 0.012  -0.57    0.     -2.81    0.      3.037   0.741   0.0076  0.0076]
```
- Send command
```
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0

ros2 topic pub --once /joint_command sensor_msgs/msg/JointState "{position: [0.0, -0.5, 0.2, 0.3, 1.0, 1.5, 0.8, 0.04, 0.04]}"
```
- output
```
Current joints: [-0.   -0.5   0.2   0.3   1.    1.5   0.8   0.04  0.04]
```
### 4. pick a cube and place it in the target
- tasks/pick_place.py
- trouble shooting
  - calibration problem
- output
```
==========================================
TASK A — PICK & PLACE
==========================================
Requested orientation: [0. 0. 1. 0.]
EE prim: /World/Franka/panda_rightfinger
Cube start: [0.5   0.    0.025]
Target: [0.4   0.3   0.025]
Calibrated grasp Z: 0.025
==========================================

[0] OPEN GRIPPER
Finger separation: 0.0800 m

[1] PRE-GRASP
  RMPFlow target: [0.5 0.  0.2]
  Gripper center: [0.4999 0.     0.2417]
  XY error: 0.0001 m

[2] DESCEND
  RMPFlow target: [0.5   0.    0.025]
  Gripper center: [ 0.5    -0.      0.0667]
  XY error: 0.0000 m

------------------------------------------
BEFORE GRASP
Cube: [0.5   0.    0.025]
Gripper center: [ 0.5    -0.      0.0667]
Finger separation: 0.0800 m
Center - cube: [-0.     -0.      0.0417]
------------------------------------------

[3] CLOSE GRIPPER

------------------------------------------
AFTER GRASP
Cube: [0.5   0.    0.025]
Gripper center: [0.5    0.     0.0667]
Finger separation: 0.0506 m
Center - cube: [-0.      0.      0.0417]
------------------------------------------

Closed finger separation: 0.0506 m

[4] LIFT
  RMPFlow target: [0.5  0.   0.25]
  Gripper center: [ 4.999e-01 -2.000e-04  2.917e-01]
  XY error: 0.0003 m
Cube before lift: [0.5   0.    0.025]
Cube after lift: [ 4.999e-01 -4.000e-04  2.500e-01]
Vertical lift: 0.2250 m

>>> GRASP CONFIRMED <<<

==========================================
MEASURED GRASP OFFSET
==========================================
Gripper center: [ 4.999e-01 -2.000e-04  2.917e-01]
Cube: [ 4.999e-01 -4.000e-04  2.500e-01]
Cube - gripper: [ 1.00e-04 -1.00e-04 -4.17e-02]
==========================================

[5] MOVE ABOVE TARGET
  RMPFlow target: [0.4  0.3  0.25]
  Gripper center: [0.3965 0.3043 0.2917]
  XY error: 0.0055 m

------------------------------------------
ABOVE TARGET
Cube: [0.3966 0.3042 0.2499]
Gripper center: [0.3965 0.3043 0.2917]
Finger separation: 0.0506 m
Center - cube: [-0.00e+00  1.00e-04  4.17e-02]
------------------------------------------


==========================================
PLACE HEIGHT CALCULATION
==========================================
Controller → center Z offset: 0.0417 m
Cube → gripper relative Z: -0.0417 m
Desired cube center Z: 0.0250 m
Desired gripper center Z: 0.0667 m
Calculated RMPFlow place Z: 0.0251 m
==========================================

[6] LOWER TO PLACE
  RMPFlow target: [0.4    0.3    0.0251]
  Gripper center: [0.4    0.3    0.0668]
  XY error: 0.0000 m

------------------------------------------
BEFORE RELEASE
Cube: [0.4    0.2997 0.0251]
Gripper center: [0.4    0.3    0.0668]
Finger separation: 0.0506 m
Center - cube: [-0.      0.0002  0.0417]
------------------------------------------

[7] RELEASE
Cube after release: [0.4    0.2997 0.0251]

[8] RETREAT
  RMPFlow target: [0.4  0.3  0.25]
  Gripper center: [0.3965 0.3044 0.2917]
  XY error: 0.0056 m


==========================================
TASK A — FINAL RESULT
==========================================
Cube start: [0.5   0.    0.025]
Cube final: [0.4    0.2997 0.0251]
Target: [0.4   0.3   0.025]

XY error: 0.0003 m
XYZ error: 0.0003 m
Tolerance: 0.0800 m

SUCCESS: True
==========================================
```