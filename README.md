# physical_ai_projects
## GPU setting
- Cloud GPU from runpod
- Task 1 ~ 8 : Issac sim 4.0 which is relatively low version. Since runpod only offers TCP, viewing the remote server screen was impossible because Issac streaming service requires UDP.
- Task 9 ~ : Issac sim 6.1.0 (found another runpod template)
<details>
<summary><strong>Server setting up for Issac sim 4.0</strong></summary>

1. Create runpod server with the template beloew
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
6. vscode + SSH setting
   - control the server with your local machine through vscode
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
7. Connect it to git to manage the repository
### Git SSH
public key
- generate a private/public key on the remote server
```
ssh-keygen -t ed25519 -C "{github_email}"
cat ~/.ssh/id_ed25519.pub
```
- add this public key on github
- ssh -T git@github.com
8. install ROS2 humble
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
```
</details>

<details>
<summary><strong>Server settung for Isaac sim 6.1.0</strong></summary>

- run a pod with the template below:
- enter server
- Vscode connection
  - ```
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    echo 'YOUR_PUBLIC_KEY' >> ~/.ssh/authorized_keys
    # type your public key
    chmod 600 ~/.ssh/authorized_keys
    ```
  - change the config of remote-ssh on your local vscode
  - ex)
  - ```
    
    ```
  - ssh runpod # on your local machine terminal
- (optinal) connect to Git
- activate the conda virtual env
  ```bash 
  source /opt/conda/etc/profile.d/conda.sh
  conda activate env_isaacsim
  ```
- set the display
  ```bash
  export DISPLAY=:1
  ```
- execute
  ```bash
  python {task}.py
  ```
</details>

# project #1: Franka pandas basics
### task1: Reading the current joint position
- 01_read_joint.py
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
### task2: move the joint position (Joint space)
02_move_joint.py
```python
# move the joints
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
### task3&4: Cartesian space + IK solver
- 03_move_cartesian_01.py
```python
ik_solver = KinematicsSolver(franka)
target = np.array([0.5, 0.4, 0.1])
action, success = ik_solver.compute_inverse_kinematics(target_position=target)
```
- output
```python
IK success : True
IK action: {'joint_positions': [-0.2474887004383594, 0.9693817698293395, 0.8151337390306523, -2.223918613446941, 2.837344281500117, 1.884653395022492, 2.2637032751245556], 'joint_velocities': None, 'joint_efforts': None}
Target: [0.5 0.4 0.1]
Actual: [0.4646386  0.37809917 0.0922472 ]
```
- 04_move_cartesian_02.py
    - compute the target joint every simulation steps based on the current robot state
```python
controller = RMPFlowController(name="franka_rmpflow_controller",
    robot_articulation=franka)
# target gripper coordinate (x, y, z)
target = np.array([0.4, 0.2, 0.5])
for _ in range(500):
    action = controller.forward(target_end_effector_position=target)
    franka.apply_action(action)
    world.step(render=False)
```
- output
```bash
Joint position: [ 0.16466147 -0.7054785   0.28579456 -2.478773    0.05316455  2.3896084
  0.7499266   0.00760282  0.00760279]
Gripper position: [0.37350965 0.20253271 0.53297734]
```

### task5: Pick-and-place
- 05_pick_and_place.py
- output

# project #2: ROS2
### task6: publish
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

### task7: . command
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
### task8: pick a cube and place it in the target
- 08_tasks/pick_place.py
- exeution
```
DISPLAY=:1 OMNI_KIT_ALLOW_ROOT=1 /isaac-sim/python.sh precise_placement.py
```
- trouble shooting
  - calibration problem
```
...
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

# project #3: Behavior cloining model
```text
physical_ai_projects/
├── 09_teleoperation/          # Control Franka Panda via ROS 2
├── 10_recoder/                # Collect and save demonstrations
│   ├── keyboard_control.py
│   ├── record_demonstrations.py
│   └── demonstrations/       # Recorded episodes (.npz)
│
├── 11_behavior_cloning/       # Train and evaluate a BC policy
│   ├── dataset.py
│   ├── chekc_dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
└── README.md
```
### task 9. Teleoperation
Implemented keyboard based teleoperation of a Franka Panda robot using ROS 2 and Isaac Sim.
### task 10. Demonstration Data 
Collected robot demonstrations, including joint states, end-effector positions, object positions, and control actions.
### task 11. Behavior cloning
Implemented and trained an MLP behavior cloning policy using PyTorch, and evaluated it in Isaac Sim.

# Project #4: Vision-Based Robot Learning