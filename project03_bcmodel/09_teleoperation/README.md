# GUI Franka Teleoperation (Isaac Sim 4.0 + ROS 2 Humble)

Two-terminal setup. Close other Isaac Sim instances before starting.

## Terminal 1: Isaac Sim GUI

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate env_isaacsim

cd /root/physical_ai_projects/09_teleoperation
export DISPLAY:=1
DISPLAY=:1 python gui_teleop.py # or python gui_teleop.py
```
Wait until World is ready
## Terminal 2: keyboard commands

```bash
cd /root/physical_ai_projects/10_recoder
python keyboard_control.py
```

Keep terminal 2 focused for key presses; watch the GUI in VNC.

W/S: X +/-; A/D: Y +/-; Q/E: Z +/-; O: open; C: close; X: quit keyboard controller.

Each movement key sends a **1 cm relative displacement**. There is no auto-repeat or continuous key-state tracking; use repeated presses. This is a first teleop milestone; it does **not yet** record demonstrations or reset episodes.
