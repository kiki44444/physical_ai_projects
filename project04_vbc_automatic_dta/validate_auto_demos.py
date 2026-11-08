import argparse
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--data-dir', default='vision_demonstrations_auto')
p.add_argument('--xy-tolerance', type=float, default=0.05)
a = p.parse_args()
files = sorted(Path(a.data_dir).glob('episode_*.npz'))
if not files:
    raise SystemExit(f'No NPZ files found in {a.data_dir}')
valid = []
for path in files:
    errors = []
    with np.load(path, allow_pickle=False) as d:
        required = ('rgb','joint_positions','ee_position','cube_position','target_position','action_delta','gripper_state','gripper_open','gripper_command','goal_position','cube_start')
        missing = [k for k in required if k not in d]
        if missing:
            print(f'[FAIL] {path.name}: missing={missing}')
            continue
        n = len(d['rgb'])
        expected = {'rgb':(n,224,224,3),'joint_positions':(n,9),'ee_position':(n,3),'cube_position':(n,3),'target_position':(n,3),'action_delta':(n,3),'gripper_open':(n,),'gripper_state':(n,),'gripper_command':(n,)}
        for key,shape in expected.items():
            if d[key].shape != shape: errors.append(f'{key} shape={d[key].shape} expected={shape}')
        if errors:
            print(f'[FAIL] {path.name}: {errors}')
            continue
        for key in ('joint_positions','ee_position','cube_position','target_position','action_delta'):
            if not np.isfinite(d[key]).all(): errors.append(f'nonfinite {key}')
        open_labels = d['gripper_open'].astype(int)
        commands = d['gripper_command'].astype(int)
        close_idx = np.flatnonzero(commands == 0)
        open_idx = np.flatnonzero(commands == 1)
        if len(close_idx) == 0 or len(open_idx) == 0 or open_idx[-1] <= close_idx[0]:
            errors.append('missing CLOSE then OPEN command')
        if not set(np.unique(open_labels)).issubset({0,1}): errors.append('invalid gripper_open labels')
        if len(close_idx) and len(open_idx):
            if open_labels[close_idx[0]] != 0 or open_labels[open_idx[-1]] != 1:
                errors.append('gripper label inconsistent with command')
        goal = d['goal_position']
        final_cube = d['cube_position'][-1]
        dist = float(np.linalg.norm(final_cube[:2] - goal[:2]))
        if dist >= a.xy_tolerance: errors.append(f'final xy={dist:.4f}')
        if abs(final_cube[2]-0.025) >= 0.012: errors.append(f'final z={final_cube[2]:.4f}')
        # Each action should correspond to the change in commanded target.
        target = d['target_position']
        delta = d['action_delta']
        alignment = float(np.max(np.abs(target[1:] - target[:-1] - delta[:-1]))) if n > 1 else float('inf')
        if alignment > 0.001: errors.append(f'action alignment max error={alignment:.5f}')
        if len(open_idx) and open_idx[-1] >= n-1: errors.append('no post-release observation')
        if errors:
            print(f'[FAIL] {path.name} frames={n}: {"; ".join(errors)}')
        else:
            valid.append(path)
            print(f'[PASS] {path.name} frames={n} xy={dist:.3f} close={close_idx[0]} open={open_idx[-1]}')
print(f'[SUMMARY] passed={len(valid)}/{len(files)}')
if len(valid) != len(files): raise SystemExit(1)
