import numpy as np
from pathlib import Path

DATA_DIR = Path("datasets/demonstrations")

files = sorted(DATA_DIR.glob("episode_*.npz"))

required_keys = [
    "time",
    "joint_positions",
    "joint_velocities",
    "ee_position",
    "ee_orientation",
    "cube_position",
    "cube_orientation",
    "action",
    "target_position",
]

print(f"Found {len(files)} episodes\n")

successes = []
lengths = []

for file in files:
    with np.load(file, allow_pickle=False) as data:
        missing = [k for k in required_keys if k not in data.files]

        if missing:
            print(f"{file.name}: Missing keys {missing}")
            continue

        n = len(data["time"])
        lengths.append(n)

        valid = all(
            np.isfinite(data[k]).all()
            for k in required_keys
        )

        actions = data["action"]
        active_actions = np.count_nonzero(
            np.any(actions != 0, axis=1)
        )

        dt = np.diff(data["time"])
        mean_dt = float(np.mean(dt)) if len(dt) else float("nan")

        cube_final = data["cube_position"][-1]
        goal = np.array([0.40, 0.30, 0.025])

        xy_error = np.linalg.norm(cube_final[:2] - goal[:2])
        z_error = abs(cube_final[2] - goal[2])

        success = bool(xy_error <= 0.05 and z_error <= 0.03)
        successes.append(success)

        print(
            f"{file.name}: "
            f"steps={n}, "
            f"dt={mean_dt:.3f}s, "
            f"active_actions={active_actions}, "
            f"finite={valid}, "
            f"success={success}, "
            f"xy_error={xy_error:.3f}m"
        )

print("\n========== SUMMARY ==========")

if lengths:
    print("Episodes:", len(lengths))
    print("Total timesteps:", sum(lengths))
    print("Mean episode length:", np.mean(lengths))

if successes:
    print("Success rate:", f"{np.mean(successes) * 100:.1f}%")

# for file in sorted(Path("datasets/demonstrations").glob("episode_*.npz")):
#     with np.load(file) as data:
#         cube = data["cube_position"][-1]
#         goal = data["target_position"][-1]

#         xy_error = np.linalg.norm(cube[:2] - goal[:2])
#         z_error = abs(cube[2] - goal[2])

#         print(
#             f"{file.name}: "
#             f"cube_z={cube[2]:.4f}, "
#             f"goal_z={goal[2]:.4f}, "
#             f"xy={xy_error:.4f}, "
#             f"z={z_error:.4f}"
#         )


#지금 할 일: Action 데이터 검증
# 현재 가장 중요한 것은 action에 저장된 값이 무엇을 의미하는지 확인하는 거야.
# import numpy as np

# data = np.load("datasets/demonstrations/episode_000.npz")

# actions = data["action"]

# print("Action shape:", actions.shape)
# print("First 10 actions:")
# print(actions[:10])

# print("\nAction min:", actions.min(axis=0))
# print("Action max:", actions.max(axis=0))

# print("\nUnique actions:")
# print(np.unique(actions, axis=0))

# print("\nNonzero ratio:",
#       np.mean(np.any(actions != 0, axis=1)))


#
# BC에서는 로봇이 상태 \(s_t\)를 관측하고 행동 \(a_t\)를 예측하도록 학습해.
# 그런데 현재 Action이 실제 움직임과 일치하는지 확인해야 해.
# 다음 코드를 실행해 봐.
# import numpy as np

# data = np.load("datasets/demonstrations/episode_000.npz")

# ee = data["ee_position"]
# actions = data["action"]

# # Actual EE displacement between consecutive samples
# actual_delta = np.diff(ee, axis=0)

# # Recorded commanded displacement
# commanded_delta = actions[:-1, :3]

# print("========== ACTION ALIGNMENT ==========")

# for axis, name in enumerate(["X", "Y", "Z"]):
#     cmd = commanded_delta[:, axis]
#     actual = actual_delta[:, axis]

#     print(f"\n{name} axis")
#     print("Command range:", cmd.min(), cmd.max())
#     print("Actual displacement range:", actual.min(), actual.max())

#     if np.std(cmd) > 1e-8 and np.std(actual) > 1e-8:
#         correlation = np.corrcoef(cmd, actual)[0, 1]
#         print("Correlation:", correlation)
#     else:
#         print("Correlation: undefined")


# ========== ACTION ALIGNMENT ==========

# X axis
# Command range: -0.010000000000000009 0.010000000000000009
# Actual displacement range: -0.0037859678268432617 0.003896564245223999
# Correlation: 0.6821354463722874

# Y axis
# Command range: -0.010000000000000009 0.020000000000000018
# Actual displacement range: -0.007624644786119461 0.014591038227081299
# Correlation: 0.6154018589974779

# Z axis
# Command range: -0.010000000000000009 0.01999999999999999
# Actual displacement range: -0.004729107022285461 0.006360188126564026
# Correlation: 0.6912806656021525
        #세 축 모두 명령한 이동 방향과 실제 End-effector 이동 사이에 양의 상관관계가 있어.
# 또한 실제 이동량이 명령량보다 작은 것은 RMPFlow가 목표 위치로 점진적으로 움직이는 제어 방식과 일치해.