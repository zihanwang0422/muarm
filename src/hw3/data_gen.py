# Import libraries
import numpy as np
import pandas as pd
from SOARM101 import SOARM101

# For reproducibility
np.random.seed(42)

# Initialize the SOARM101 class
arm = SOARM101()

################################################################################
# Parameters
################################################################################

# Number of reference samples to generate (each will be perturbed)
num_samples = 5000
# 2 degrees of perturbation on each side
perturbation_range = np.deg2rad(2)
# Number of times to perturb each reference joint angle set
perturbation_iterations = 8

# Define joint angle limits (converted from degrees to radians)
joint_limits = [
    (np.deg2rad(-30), np.deg2rad(30)),  # Joint 1: -30° to 30°
    (np.deg2rad(45), np.deg2rad(90)),  # Joint 2: 45° to 90°
    (np.deg2rad(-120), np.deg2rad(-90)),  # Joint 3: -120° to -90°
    (np.deg2rad(-90), np.deg2rad(-60)),  # Joint 4: -90° to -60°
    (np.deg2rad(-45), np.deg2rad(45)),  # Joint 5: -45° to 45°
    (np.deg2rad(-20), np.deg2rad(10)),  # Joint 6: -20° to 10°
]

print(f"Generating {num_samples} samples...")

# Initialize lists to store data
target_joint_angles_list = []
current_joint_angles_list = []
quaternions_list = []
positions_list = []

# Precompute bounds for clipping
lower_bounds = np.array([lim[0] for lim in joint_limits])
upper_bounds = np.array([lim[1] for lim in joint_limits])

# Generate dataset
for _ in range(num_samples):
    # Step 1: random reference joint angles within limits
    reference_angles = np.array(
        [np.random.uniform(low, high) for low, high in joint_limits]
    )

    for _ in range(perturbation_iterations):
        # Step 2: perturb reference angles and clip to limits
        perturbation = np.random.uniform(
            -perturbation_range, perturbation_range, size=len(joint_limits)
        )
        target_angles = np.clip(reference_angles + perturbation, lower_bounds, upper_bounds)

        # Step 3: forward kinematics for target angles
        quaternion, position = arm.forward_kinematics(target_angles)

        # Collect data rows
        target_joint_angles_list.append(target_angles)
        current_joint_angles_list.append(reference_angles)
        quaternions_list.append(quaternion)
        positions_list.append(position)

# Step 3: Make a DataFrame with the data
# Dataset Structure should be like this:
# ==============================================================
# INPUTS (13 features):
#   - reference_joint_1 to reference_joint_6 (6): reference joint angles
#   - target_quat_x/y/z/w (4): target orientation
#   - target_pos_x/y/z (3): target position
#
# OUTPUTS (6 targets):
#   - target_joint_1 to target_joint_6: joint angles to reach target
##########################################################
# TODO: Please add the code here
##########################################################
# Build DataFrame in required order
data = {
    **{f"reference_joint_{i+1}": [angles[i] for angles in current_joint_angles_list] for i in range(6)},
    "target_quat_x": [quat[0] for quat in quaternions_list],
    "target_quat_y": [quat[1] for quat in quaternions_list],
    "target_quat_z": [quat[2] for quat in quaternions_list],
    "target_quat_w": [quat[3] for quat in quaternions_list],
    "target_pos_x": [pos[0] for pos in positions_list],
    "target_pos_y": [pos[1] for pos in positions_list],
    "target_pos_z": [pos[2] for pos in positions_list],
    **{f"target_joint_{i+1}": [angles[i] for angles in target_joint_angles_list] for i in range(6)},
}

df = pd.DataFrame(data)

# Step 4: Save the DataFrame to a CSV file
csv_filename = "ik_dataset.csv"
##########################################################
# TODO: Please add the code here
##########################################################
df.to_csv(csv_filename, index=False)

# Basic sanity print
print(df.head())
print(f"Saved dataset with {len(df)} rows and {len(df.columns)} columns.")


print(f"Dataset saved to {csv_filename}")
