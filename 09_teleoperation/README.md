# Phase 4 — GUI Franka Teleoperation (Isaac Sim 4.0 + ROS 2 Humble)

Two-terminal setup on the **same RunPod container**. Close other Isaac Sim instances before starting.

## Terminal 1: Isaac Sim GUI

```bash
cd /workspace/physical_ai_projects/09_teleoperation
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
DISPLAY=:1 OMNI_KIT_ALLOW_ROOT=1 /isaac-sim/python.sh gui_teleop.py
```

## Terminal 2: keyboard commands

```bash
cd /workspace/physical_ai_projects/09_teleoperation
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
python3 teleop/keyboard_control.py
```

Keep terminal 2 focused for key presses; watch the GUI in noVNC.

W/S: X +/-; A/D: Y +/-; Q/E: Z +/-; O: open; C: close; X: quit keyboard controller.

Each movement key sends a **1 cm relative displacement**. There is no auto-repeat or continuous key-state tracking; use repeated presses. This is a first teleop milestone; it does **not yet** record demonstrations or reset episodes.

If `rclpy` or `geometry_msgs` cannot be imported by Isaac Sim's Python, check the sourced ROS environment and Python version compatibility before proceeding. If GUI starts but the robot does not move, inspect the GUI terminal for errors and confirm the two processes see the same ROS domain and RMW implementation.
