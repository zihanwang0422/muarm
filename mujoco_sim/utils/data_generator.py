"""
IK Dataset Generator for Neural Network Training.

Generates (reference_angles, target_pose) → target_angles pairs
using FK with perturbation, adapted for the 5-DOF arm.
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kinematics import ForwardKinematics


def generate_ik_dataset(
    num_samples: int = 5000,
    perturbation_deg: float = 2.0,
    perturbation_iters: int = 8,
    joint_limits_deg=None,
    seed: int = 42,
    output_csv: str = "ik_dataset.csv",
) -> pd.DataFrame:
    """
    Generate IK training data for the 5-DOF arm.

    Output columns:
      Inputs  (12): reference_joint_1..5, target_quat_w/x/y/z, target_pos_x/y/z
      Targets (5):  target_joint_1..5
    """
    np.random.seed(seed)
    fk = ForwardKinematics()
    pert_rad = np.deg2rad(perturbation_deg)

    if joint_limits_deg is None:
        # Reasonable manipulation workspace subset
        joint_limits_deg = [
            (-60, 60),     # Rotation
            (-120, -30),   # Pitch
            (30, 150),     # Elbow
            (-60, 60),     # Wrist_Pitch
            (-90, 90),     # Wrist_Roll
        ]

    jl = [(np.deg2rad(lo), np.deg2rad(hi)) for lo, hi in joint_limits_deg]
    lower = np.array([l[0] for l in jl])
    upper = np.array([l[1] for l in jl])

    rows = []
    for _ in range(num_samples):
        ref = np.array([np.random.uniform(lo, hi) for lo, hi in jl])
        for _ in range(perturbation_iters):
            pert = np.random.uniform(-pert_rad, pert_rad, 5)
            target = np.clip(ref + pert, lower, upper)
            pos, quat = fk.compute(target)  # quat is wxyz
            rows.append(np.concatenate([ref, quat, pos, target]))

    cols = (
        [f"reference_joint_{i+1}" for i in range(5)]
        + ["target_quat_w", "target_quat_x", "target_quat_y", "target_quat_z"]
        + ["target_pos_x", "target_pos_y", "target_pos_z"]
        + [f"target_joint_{i+1}" for i in range(5)]
    )
    df = pd.DataFrame(rows, columns=cols)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} samples → {output_csv}")
    return df


if __name__ == "__main__":
    df = generate_ik_dataset()
    print(df.head())
