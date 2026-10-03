# physical_ai_projects
### vscode + SSH
1. install remote SSH extension
2. public key of mine
```
Get-Content ~\.ssh\id_ed25519.pub
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
- ssh -T {github_email}
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
1. Franka + ground plane + cube
2. stardalone python script (/workspace/main.py)
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
3. Reading the current joint position
```python
from omni.isaac.kit import SimulationApp

# Start Isaac Sim with GUI
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.franka import Franka

# Create world
# 1 USD unit = 1 meter
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Add franka to the world
franka = world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="franka",
    )
)

world.reset()

# Read the current position
joint_positions = franka.get_joint_positions()
print(joint_positions)

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
```
- Terminal output
```python
[ 0.012  -0.570  0.     -2.81  0.     3.037  0.741  0.0076  0.0076 ]
   ↑       ↑      ↑       ↑     ↑      ↑      ↑       ↑       ↑
 joint1  joint2  joint3  joint4 joint5 joint6 joint7  finger  finger
```
4. Move the joint position (Joint space)
```python
# move the joints
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np

# radian
q = np.array([0.0, -0.5, 0.0, -2.0 , 0.0, 1.5, 0.7, 0.04, 0.04])
action = ArticulationAction(joint_positions=q)
franka.apply_action(action)
```
5. Cartesian space + IK solver
