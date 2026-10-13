# Phase 4 — Pick & Place demonstrations (Isaac Sim 4.0 / ROS 2 Humble)

Files: `gui_teleop.py`, `keyboard_control.py`.

Start Isaac Sim in terminal 1:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
DISPLAY=:1 OMNI_KIT_ALLOW_ROOT=1 /isaac-sim/python.sh gui_teleop.py
```

Start keyboard in terminal 2 (same ROS env):

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
python3 keyboard_control.py
```

Keys: W/S X, A/D Y, Q/E Z, O open, C close, N reset and randomize cube, R start recording, P stop/save/evaluate, X quit keyboard.

Task: move 5 cm cube from randomized XY position to green marker at `[0.40, 0.30, 0.025]` meters. `success` checks final XY error <= 5 cm and Z error <= 3 cm. Output `datasets/demonstrations/episode_###.npz` includes per-step observations/actions and episode-level success, final errors, goal and randomized start.

Notes: keep the keyboard terminal focused. `N` is accepted only when not recording. Recording samples are at 10 Hz *simulation time*. The action is accumulated EE target delta since last sample plus gripper-open state. This is a prototype: further work is needed for precise observation-action alignment, success after physical settling, and robust resets for policy training. Test the robot first before collecting large datasets.
