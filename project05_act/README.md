## Experimental Results

We evaluated two vision-based manipulation policies for a Franka Panda pick-and-place task in NVIDIA Isaac Sim 6.1.

### Models

- **Vision BC:** Pretrained ResNet18 + MLP, predicting a single action per observation.
- **Action Chunking Transformer (ACT-style):** ResNet18 + Transformer, predicting 20 future actions per observation.
- **Dataset:** Approximately 100 successful expert demonstrations.
- **Observation:** RGB image (224 × 224) and 12-dimensional robot state.
- **Action:** 3D end-effector target displacement and binary gripper command.

### Closed-Loop Evaluation

| Metric | Vision BC | ACT (Chunk) | ACT (Receding) |
|---|---:|---:|---:|
| Success Rate | 20% | 40% | 40% |
| Successful Episodes | 1/5 | 2/5 | 2/5 |
| Mean Final XY Error | 14.2 cm | 10.1 cm | 15.8 cm |
| Episode Duration Limit | 40 s | 60 s | 60 s |

### Key Findings

- Both action-chunking execution strategies achieved a 40% pick-and-place success rate in preliminary evaluations.
- Chunk execution achieved a lower mean final XY error than receding-horizon execution.
- Increasing the expert demonstration dataset from 20 to approximately 100 episodes reduced the Vision BC offline position MAE from 0.928 mm to 0.488 mm (~47% improvement).

### Limitations

Results are preliminary due to the small evaluation sample (5 episodes per configuration). BC and ACT were evaluated with different time limits, so their success rates are not directly comparable under identical conditions.

The ACT-style model uses deterministic Transformer-based action chunking rather than the original CVAE-based ACT architecture.

### Conclusion

This project demonstrates an end-to-end vision-based robotic manipulation pipeline, including automated expert demonstration collection, imitation learning, Transformer-based action chunking, and closed-loop evaluation in Isaac Sim.