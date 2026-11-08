# Vision Behavior Cloning for Franka Pick-and-Place in Isaac Sim

## Task

1. Observe the robot and a cube at a randomized starting position.
2. Move the end effector toward the cube (approach).
3. Descend and close the gripper (grasp).
4. Lift the cube and move it toward the goal (transport).
5. Lower and release the cube (place).

## Environment

- NVIDIA Isaac Sim **6.1.0**
- Franka Panda robot with RMPFlow motion control
- Python / PyTorch / torchvision / NumPy
- GPU experiments on an NVIDIA RTX 4090 (RunPod)
- Simulation physics step: approximately **1/60 s**
- Policy control frequency: **10 Hz** (one action every six physics steps)
- RGB camera images: **224 × 224 × 3**

## Project files

The following scripts were used during development (file names correspond to the working project; this README does not bundle their source code):

```text
.
├── generate_expert_demos.py       # Automatic expert trajectories and NPZ recording
├── vbc_model.py                   # VisionBehaviorCloning network
├── train_auto_12d.py              # Training and episode-level validation split
├── offline_test_12d.py            # Teacher-forced offline action evaluation
├── evaluate_auto_12d.py          # Closed-loop Isaac Sim evaluation
├── vision_demonstrations_auto/    # episode_*.npz files (generated)
├── vision_bc_12d_auto_best.pt     # Baseline checkpoint (~20 episodes)
└── vision_bc_12d_auto_100ep.pt    # Expanded-data checkpoint (~100 episodes)
```

## Expert demonstration collection

A scripted expert uses RMPFlow to control the robot through approach, descent, gripper close, lift, transport, descent, release, and retreat. Only successful demonstrations are saved.

Each recorded frame includes fields such as:

| Field | Description |
| --- | --- |
| `rgb` | Camera image, shape `(224, 224, 3)` |
| `joint_positions` | 9 joint values: 7 arm joints + 2 finger joints |
| `ee_position` | End-effector world position, 3 values |
| `target_position` | Current RMPFlow target position, 3 values; recorded but **not** used in the 12D policy input |
| `cube_position` | Cube world position |
| `action_delta` | Change in the **RMPFlow target position**, 3 values |
| `gripper_open` | Desired gripper state after the command: 1 = open, 0 = closed |

**Important action semantics:** `action_delta` is the change in the controller target, **not** the change in measured end-effector position:

```python
expert_action_delta = next_target - previous_target
next_target = previous_target + predicted_action_delta
```

Measured end-effector coordinates can differ from RMPFlow target coordinates; do not interchange them when initializing or updating the controller.

Example collection command (adjust flags to the version of the script in your checkout):

```bash
DISPLAY=:1 OMNI_KIT_ALLOW_ROOT=1 python -u generate_expert_demos.py \
  --episodes 80 \
  --max-attempts 400 \
  --seed 123 \
  --output-dir vision_demonstrations_auto
```

The expanded experiment added demonstrations to the initial collection, targeting approximately 100 successful episodes. Verify the actual number of saved `episode_*.npz` files before training.

## Model architecture

**Input:** an RGB image and a 12-dimensional proprioceptive state:

```text
RGB image (224 × 224 × 3)
    └── ImageNet-pretrained ResNet-18, final FC replaced with Identity
         └── 512-dimensional visual feature

Robot state (12D = 9 joint positions + 3 EE world-position values)
    └── Linear(12, 64) + ReLU
         └── 64-dimensional state feature

Concatenate → 576D
    └── Linear(576, 256) + ReLU
    └── Linear(256, 128) + ReLU
         ├── Linear(128, 3): target-position delta
         └── Linear(128, 1): gripper logit
```

The ResNet-18 backbone is **frozen** during training and kept in evaluation mode. Images use `ResNet18_Weights.DEFAULT.transforms()` for preprocessing. The gripper output is interpreted through a sigmoid: probability ≥ 0.5 means **open**; otherwise **closed**.

### Loss

```python
target_delta_scaled = action_delta * 100.0
position_loss = mse(predicted_delta_scaled, target_delta_scaled)
gripper_loss = binary_cross_entropy_with_logits(gripper_logits, gripper_open)
total_loss = position_loss + 2.0 * gripper_loss
```

At inference, divide the predicted delta by the checkpoint's `position_scale` (100.0) before updating the RMPFlow target. The reported 100-episode experiment used the **original unweighted MSE + BCE loss**, without approach-specific sample weighting.

## Training

Training uses an **episode-level** train/validation split to avoid mixing frames from the same demonstration across the two sets. The default validation fraction is 20%, batch size 32, learning rate `1e-4`, and maximum 30 epochs. The checkpoint with the lowest validation total loss is saved.

```bash
python train_auto_12d.py \
  --data-dir vision_demonstrations_auto \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --val-fraction 0.2 \
  --position-scale 100 \
  --gripper-weight 2 \
  --output vision_bc_12d_auto_100ep.pt
```

The training checkpoint stores model weights, input dimension, scaling, frozen-backbone setting, best epoch/loss, and the train/validation episode filenames.

## Offline evaluation

Offline evaluation runs the policy on recorded expert observations and compares its predicted actions with the recorded expert actions. This is **teacher-forced evaluation**, not a closed-loop simulation rollout.

```bash
python offline_test_12d.py \
  --checkpoint vision_bc_12d_auto_100ep.pt \
  --data-dir vision_demonstrations_auto \
  --episodes 3
```

### Results: approximately 20 vs. 100 demonstrations

The following measurements were obtained on the same three selected recordings (986 frames in total):

| Metric | ~20 episodes | ~100 episodes |
| --- | ---: | ---: |
| Position MSE (m²) | 2.48 × 10⁻⁶ | **0.70 × 10⁻⁶** |
| Position MAE | 0.928 mm | **0.488 mm** |
| X-axis MAE | 0.416 mm | **0.353 mm** |
| Y-axis MAE | 0.824 mm | **0.456 mm** |
| Z-axis MAE | 1.542 mm | **0.653 mm** |
| X direction accuracy | 60.6% | **97.8%** |
| Y direction accuracy | 97.1% | 96.4% |
| Z direction accuracy | 95.8% | **99.3%** |
| Gripper accuracy | 93.7% | **99.3%** |

Direction accuracy is measured only where the corresponding expert action magnitude exceeds `0.0005 m` (137 X samples, 137 Y samples, 408 Z samples). The new model reduced overall MAE by approximately **47%**.

An illustrative first-frame X-action comparison (mm per control step):

| Example | Expert | ~20-episode model | ~100-episode model |
| --- | ---: | ---: | ---: |
| Episode A | +6.57 | -0.11 | +2.12 |
| Episode B | -4.87 | +0.05 | -1.35 |
| Episode C | -7.44 | -0.20 | -4.17 |

The expanded-data model predicts the initial approach direction more reliably, although action magnitudes are still underestimated in these examples.

**Evaluation caveat:** These three recordings were not established as an independent held-out test set for the 100-episode model. The offline numbers demonstrate improved prediction on the selected recordings, but do **not** by themselves establish out-of-sample generalization.

## Closed-loop evaluation

The trained policy is evaluated in Isaac Sim with randomized cube starting positions. At each control step it receives a fresh RGB image and 12D robot state, predicts a target delta and gripper command, and updates the robot controller.

```bash
DISPLAY=:1 OMNI_KIT_ALLOW_ROOT=1 python -u evaluate_auto_12d.py \
  --checkpoint vision_bc_12d_auto_100ep.pt \
  --episodes 5 \
  --duration 40 \
  --seed 2026 \
  --log-every 10
```

### Initial five-episode rollout

| Metric | Result |
| --- | ---: |
| Episodes | 5 |
| Gripper closed at least once | 3/5 |
| Cube lifted | 2/5 |
| Gripper reopened at least once | 3/5 |
| Successful stable placement | **1/5 (20%)** |
| Mean final cube-to-goal XY error | **0.142 m** |

| Episode | Outcome |
| --- | --- |
| 1 | Approached/descended, but gripper never closed; no lift |
| 2 | Grasped and lifted the cube, transported toward the goal, but did not finish placement within 40 s |
| 3 | Descended but did not close the gripper; no lift |
| 4 | Repeated gripper switching and cube displacement; no lift |
| 5 | **Successful pick, lift, transport, release, and stable placement** |

For successful **Episode 5**, the log shows:

- Step **92**: gripper closes.
- Step **110**: cube is lifted (`z ≈ 0.112 m`).
- Step **180**: cube is near the goal (`xy error ≈ 0.003 m`) while still lifted.
- Step **217**: gripper opens to release the cube.
- Step **282**: stable placement success; final cube-to-goal XY error **0.028 m**.

These observations establish a working end-to-end behavior in simulation, but **5 trials are insufficient for a reliable success-rate estimate**. Gripper release is inferred from gripper commands and cube pose, not measured contact forces.

### Reproducibility note: controller initialization

The expert generator used an initial RMPFlow target of approximately **`[0.50, 0.00, 0.20]`**, whereas the recorded evaluation log's first predicted target is around **`[0.50, 0.04, 0.24]`**. This suggests the evaluated script used a different initial controller target. **Confirm and align the expert and evaluator target initialization before making a controlled performance comparison.** The measured EE pose (around `[0.497, 0.086, 0.280]` in the log) is not interchangeable with either controller target.

## Limitations and next steps

1. **Improve grasp reliability:** investigate failures to close the gripper and rapid open/close oscillations; consider command hysteresis or a minimum dwell time, then measure the impact.
2. **Evaluate longer horizons:** test 60-second episodes; some runs lifted and transported the cube but did not complete placement within 40 seconds.
3. **Measure generalization:** reserve unseen demonstration episodes and test on many independent cube starts; report success rate with the trial count.
4. **Check state observability:** the 12D state excludes the RMPFlow target and explicit task phase, which may make certain actions ambiguous from a single observation.
5. **Consider visual fine-tuning and more diverse demonstrations:** the current visual backbone is frozen; evaluate improvements systematically rather than changing data, model, and loss simultaneously.
6. **Align evaluation settings:** verify camera pose, image preprocessing, controller initialization, control frequency, and checkpoint scaling match the demonstration/training setup.

## Current status

- [x] Automatic expert demonstration generation
- [x] 12D vision + robot-state BC model
- [x] Supervised training with episode-level validation
- [x] Expansion from ~20 to ~100 demonstrations
- [x] Offline baseline-versus-expanded-data comparison
- [x] Closed-loop evaluation in Isaac Sim
- [x] First successful simulated pick-and-place
- [ ] Reliable, repeatable grasping and placement
- [ ] Independent held-out benchmark with more trials
- [ ] Fully pinned reproducible environment

---

**Project status:** Proof of concept demonstrated in Isaac Sim; robustness and generalization remain ongoing work.
