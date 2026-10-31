## Terminal 1: Isaac Sim GUI

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate env_isaacsim

cd /root/physical_ai_projects/09_teleoperation
export DISPLAY:=1
DISPLAY=:1 python gui_teleop.py # or python gui_teleop.py
```

## Terminal 2: keyboard commands
```bash
source /opt/ros/humble/setup.bash
cd /root/Desktop/physical_ai_projects/10_recoder
/usr/bin/python3 keyboard_control.py
```
