"""
IK Dataset Generator for Neural Network Training.

Generates (reference_angles, target_pose) → target_angles pairs
using forward kinematics with perturbation, identical to the original
hw3/data_gen.py but decoupled from ROS.
"""

import numpy as np
import pandas as pd
from ..kinematics import ForwardKinematics


def generate_ik_dataset(
    num_samples: int = 5000,
    perturbation_deg: float = 2.0,
    perturbation_iters: int = 8,
    joint_limits_deg: list = None,
    seed: int = 42,
    output_csv: str = "ik_dataset.csv",
) -> pd.DataFrame:
    """
    Generate an IK training dataset.

    For each random reference configuration, applies small perturbations
    and records (reference_joints, target_pose, target_joints).

    Args:
        num_samples: Number of base reference samples.
        perturbation_deg: Perturbation range in degrees.
        perturbation_iters: Perturbations per reference sample.
        joint_limits_deg: List of (min_deg, max_deg) for each joint.
        seed: Random seed.
        output_csv: Output CSV path (None to skip saving).

    Returns:
        DataFrame with 13 input features and 6 target joint angles.
    """
    np.random.seed(seed)
    fk = ForwardKinematics()
    perturbation_rad = np.deg2rad(perturbation_deg)

    if joint_limits_deg is None:
        joint_limits_deg = [
            (-30, 30), (45, 90), (-120, -90),
            (-90, -60), (-45, 45), (-20, 10),
        ]

    joint_limits = [(np.deg2rad(lo), np.deg2rad(hi)) for lo, hi in joint_limits_deg]
    lower = np.array([l[0] for l in joint_limits])
    upper = np.array([l[1] for l in joint_limits])

    rows = []
    for _ in range(num_samples):
        ref = np.array([np.random.uniform(lo, hi) for lo, hi in joint_limits])
        for _ in range(perturbation_iters):
            pert = np.random.uniform(-perturbation_rad, perturbation_rad, 6)
            target_angles = np.clip(ref + pert, lower, upper)
            quat, pos = fk.compute(target_angles)
            rows.append(np.concatenate([ref, quat, pos, target_angles]))

    cols = (
        [f"reference_joint_{i+1}" for i in range(6)]
        + ["target_quat_x", "target_quat_y", "target_quat_z", "target_quat_w"]
        + ["target_pos_x", "target_pos_y", "target_pos_z"]
        + [f"target_joint_{i+1}" for i in range(6)]
    )
    df = pd.DataFrame(rows, columns=cols)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} samples to {output_csv}")

    return df


if __name__ == "__main__":
    df = generate_ik_dataset()
    print(df.head())
    print(f"Shape: {df.shape}")
