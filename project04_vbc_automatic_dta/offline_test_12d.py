
import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.models import ResNet18_Weights

from vbc_model import VisionBehaviorCloning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="vision_demonstrations_auto"
    )
    parser.add_argument(
        "--checkpoint",
        default="vision_bc_12d_auto_best.pt"
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--position-scale", type=float, default=100.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device

    files = sorted(Path(args.data_dir).glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No episode_*.npz files in {args.data_dir}"
        )

    files = files[:args.episodes]

    model = VisionBehaviorCloning(
        state_dim=12,
        pretrained=False,
        freeze_backbone=True
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True
    )

    # Support plain state_dict and wrapped checkpoints.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    transform = ResNet18_Weights.DEFAULT.transforms()

    all_true = []
    all_pred = []
    all_grip_true = []
    all_grip_pred = []

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {len(files)}")

    with torch.inference_mode():
        for file in files:
            print(f"\n{'=' * 60}")
            print(file.name)

            with np.load(file) as data:
                images = data["rgb"]
                joints = data["joint_positions"]
                ee = data["ee_position"]
                actions = data["action_delta"]

                # gripper_open is the action label.
                grip_labels = data["gripper_open"]

                n = len(actions)

                episode_true = []
                episode_pred = []

                for i in range(n):
                    image = torch.from_numpy(
                        np.ascontiguousarray(images[i])
                    ).permute(2, 0, 1)

                    image = transform(image).unsqueeze(0).to(device)

                    state = np.concatenate(
                        [joints[i], ee[i]]
                    ).astype(np.float32)

                    if state.shape != (12,):
                        raise ValueError(
                            f"Expected 12D state, got {state.shape}"
                        )

                    state = torch.from_numpy(
                        state
                    ).unsqueeze(0).to(device)

                    pred_position, pred_gripper = model(
                        image, state
                    )

                    predicted_delta = (
                        pred_position[0].cpu().numpy()
                        / args.position_scale
                    )

                    gripper_prob = torch.sigmoid(
                        pred_gripper
                    ).item()

                    gripper_prediction = int(
                        gripper_prob >= 0.5
                    )

                    true_delta = actions[i]

                    episode_true.append(true_delta)
                    episode_pred.append(predicted_delta)

                    all_grip_true.append(int(grip_labels[i]))
                    all_grip_pred.append(gripper_prediction)

                    if i < 10:
                        print(
                            f"[FRAME {i:03d}] "
                            f"Expert={np.round(true_delta, 5)} "
                            f"Pred={np.round(predicted_delta, 5)} "
                            f"Grip true={int(grip_labels[i])} "
                            f"pred={gripper_prediction} "
                            f"prob={gripper_prob:.3f}"
                        )

                episode_true = np.asarray(episode_true)
                episode_pred = np.asarray(episode_pred)

                all_true.append(episode_true)
                all_pred.append(episode_pred)

                mse = np.mean(
                    (episode_true - episode_pred) ** 2
                )
                mae = np.mean(
                    np.abs(episode_true - episode_pred)
                )

                print(f"\nFrames: {n}")
                print(f"Position MSE: {mse:.8f}")
                print(f"Position MAE: {mae:.6f} m")

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    all_grip_true = np.asarray(all_grip_true)
    all_grip_pred = np.asarray(all_grip_pred)

    print(f"\n{'=' * 60}")
    print("OVERALL OFFLINE EVALUATION")
    print(f"{'=' * 60}")

    mse = np.mean((all_true - all_pred) ** 2)
    mae = np.mean(np.abs(all_true - all_pred))

    print(f"Position MSE: {mse:.8f}")
    print(f"Position MAE: {mae:.6f} m")

    print("\nPer-axis MAE:")
    for axis, idx in zip("XYZ", range(3)):
        error = np.mean(
            np.abs(all_true[:, idx] - all_pred[:, idx])
        )
        print(f"  {axis}: {error:.6f} m")

    # Ignore near-zero actions for direction accuracy.
    print("\nDirection accuracy (|Expert delta| > 0.0005m):")
    for axis, idx in zip("XYZ", range(3)):
        mask = np.abs(all_true[:, idx]) > 0.0005

        if mask.sum() == 0:
            print(f"  {axis}: N/A")
            continue

        accuracy = np.mean(
            np.sign(all_true[mask, idx])
            == np.sign(all_pred[mask, idx])
        )
        print(
            f"  {axis}: {accuracy * 100:.1f}% "
            f"({mask.sum()} samples)"
        )

    gripper_accuracy = np.mean(
        all_grip_true == all_grip_pred
    )

    print(
        f"\nGripper accuracy: "
        f"{gripper_accuracy * 100:.1f}%"
    )

    print("\n[NOTE]")
    print("This is teacher-forced offline evaluation.")
    print("It does not measure closed-loop pick/place success.")
    print("For generalization, evaluate held-out episodes.")


if __name__ == "__main__":
    main()
