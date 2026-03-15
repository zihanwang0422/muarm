"""Example: Panda FK/IK verification using the kinematic module.

Demonstrates:
- Loading the Panda robot model
- Computing forward kinematics
- Solving inverse kinematics for a target pose
- Verifying FK(IK(target)) ≈ target
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from kinematic import PandaKinematics
from utils import transform2mat, mat2transform

# Build kinematics from MJCF
kin = PandaKinematics(ee_frame="link7")
kin.build_from_mjcf("models/franka_emika_panda/panda.xml")

# Define a target pose
x, y, z = 0.5, 0.0, 0.3
roll, pitch, yaw = np.pi, 0.0, 0.0
target_tf = transform2mat(x, y, z, roll, pitch, yaw)
print(f"Target pose: x={x}, y={y}, z={z}, roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")

# Solve IK
q_sol, info = kin.ik(target_tf)
print(f"IK success: {info['success']}")
print(f"Joint angles: {np.round(q_sol, 4)}")

# Verify with FK
T_fk = kin.fk(q_sol)
x2, y2, z2, r2, p2, y2_ = mat2transform(T_fk)
print(f"\nFK verification:")
print(f"  Position: [{x2:.4f}, {y2:.4f}, {z2:.4f}]")
print(f"  Euler:    [{r2:.4f}, {p2:.4f}, {y2_:.4f}]")

pos_error = np.linalg.norm(T_fk[:3, 3] - target_tf[:3, 3])
print(f"\nPosition error: {pos_error:.6f} m")
