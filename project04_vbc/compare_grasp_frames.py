import numpy as np
from pathlib import Path
from PIL import Image

DATA_DIR = Path("vision_demonstrations")
OUTPUT_DIR = Path("grasp_frame_comparison")
OUTPUT_DIR.mkdir(exist_ok=True)

files = sorted(DATA_DIR.glob("episode_*.npz"))

if not files:
    raise FileNotFoundError("No demonstration files found")

file = files[0]

with np.load(file) as d:
    # Load observations
    rgb = d["rgb"]
    joints = d["joint_positions"]
    ee = d["ee_position"]
    cube = d["cube_position"]

    # Load actions
    delta = d["action_delta"]
    gripper_open = d["gripper_open"]
    gripper_state = d["gripper_state"]
    gripper_command = d["gripper_command"]

    # 10D state: 7 arm joints + 3 EE positions
    state = np.concatenate(
        [joints[:, :7], ee],
        axis=1
    )

    assert state.shape[1] == 10

    f1, f2 = 66, 67

    # 1. State comparison
    s1 = state[f1]
    s2 = state[f2]

    state_diff = s2 - s1

    print("\n========== STATE COMPARISON ==========")

    print(f"Frame {f1} state:", np.round(s1, 6))
    print(f"Frame {f2} state:", np.round(s2, 6))

    print("\nState difference:")
    print(np.round(state_diff, 8))

    print("\nState L2 distance:")
    print(np.linalg.norm(state_diff))

    print("\nArm joint L2 distance:")
    print(np.linalg.norm(joints[f2, :7] - joints[f1, :7]))

    print("\nEE position difference (m):")
    print(ee[f2] - ee[f1])

    print("EE distance (mm):")
    print(np.linalg.norm(ee[f2] - ee[f1]) * 1000)

    # 2. RGB comparison
    img1 = rgb[f1].astype(np.float32)
    img2 = rgb[f2].astype(np.float32)

    diff = np.abs(img2 - img1)

    mae = diff.mean()
    rmse = np.sqrt(np.mean((img2 - img1) ** 2))
    max_diff = diff.max()

    print("\n========== RGB COMPARISON ==========")

    print("RGB MAE:", mae)
    print("RGB RMSE:", rmse)
    print("RGB max difference:", max_diff)
    print("Changed pixel channel percentage (%):",
          100 * np.mean(diff > 0))

    # Save comparison images
    Image.fromarray(rgb[f1]).save(
        OUTPUT_DIR / "frame_066.png"
    )

    Image.fromarray(rgb[f2]).save(
        OUTPUT_DIR / "frame_067.png"
    )

    # Difference visualization
    diff_vis = np.clip(diff * 5, 0, 255).astype(np.uint8)

    Image.fromarray(diff_vis).save(
        OUTPUT_DIR / "difference_x5.png"
    )

    # Side-by-side comparison
    comparison = np.concatenate(
        [rgb[f1], rgb[f2], diff_vis],
        axis=1
    )

    Image.fromarray(comparison).save(
        OUTPUT_DIR / "comparison.png"
    )

    # 3. Action comparison
    print("\n========== ACTION COMPARISON ==========")

    for f in [f1, f2]:
        print(f"\nFrame {f}")
        print("EE:", np.round(ee[f], 6))
        print("Cube:", np.round(cube[f], 6))
        print("Action delta:", delta[f])
        print("Gripper OPEN label:", gripper_open[f])
        print("Gripper state:", gripper_state[f])
        print("Gripper command:", gripper_command[f])

    print("\n========== CONCLUSION ==========")

    if np.linalg.norm(state_diff) < 1e-3 and mae < 2.0:
        print(
            "Observations are very similar, "
            "but the gripper action may differ."
        )
        print(
            "This suggests the policy may need "
            "temporal or phase information."
        )
    else:
        print(
            "Observations differ measurably. "
            "Further investigation is needed."
        )

    print("\nSaved images to:", OUTPUT_DIR.resolve())