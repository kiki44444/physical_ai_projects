import numpy as np
from pathlib import Path

file = sorted(Path("vision_demonstrations").glob("episode_*.npz"))[0]

with np.load(file) as d:
    ee = d["ee_position"]
    cube = d["cube_position"]
    target = d["target_position"]
    delta = d["action_delta"]

    gripper_open = d["gripper_open"]
    gripper_state = d["gripper_state"]
    gripper_command = d["gripper_command"]

    print("FILE:", file)
    print("FRAMES:", len(ee))

    print("\nEXPERT FRAMES 55–85")
    print("-" * 140)

    for i in range(55, min(86, len(ee))):
        print(
            f"Frame {i:03d} | "
            f"EE={np.round(ee[i], 4)} | "
            f"Target={np.round(target[i], 4)} | "
            f"Cube={np.round(cube[i], 4)} | "
            f"Delta={np.round(delta[i], 5)} | "
            f"OpenLabel={gripper_open[i]} | "
            f"State={gripper_state[i]} | "
            f"Command={gripper_command[i]}"
        )

    print("\nGRIPPER LABEL TRANSITIONS")
    print("-" * 80)

    transitions = np.where(
        np.diff(gripper_open.astype(int)) != 0
    )[0] + 1

    for i in transitions:
        print(
            f"Frame {i:03d}: "
            f"{gripper_open[i-1]} -> {gripper_open[i]} | "
            f"EE={np.round(ee[i], 4)} | "
            f"Target={np.round(target[i], 4)} | "
            f"Cube={np.round(cube[i], 4)}"
        )

    print("\nINITIAL EXPERT STATE")
    print("EE:", ee[0])
    print("Target:", target[0])
    print("Cube:", cube[0])
    print("Target - EE:", target[0] - ee[0])

    print("\nEXPERT GRASP STATE (FRAME 67)")
    i = 67
    print("EE:", ee[i])
    print("Target:", target[i])
    print("Cube:", cube[i])
    print("Target - EE:", target[i] - ee[i])
    print("Gripper open:", gripper_open[i])
    print("Gripper state:", gripper_state[i])
    print("Gripper command:", gripper_command[i])