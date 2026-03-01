#!/usr/bin/env python3
"""
SO-ARM100 MuJoCo Simulation — Main Entry Point

Demonstrates model-based manipulation:
  1. FK verification (custom FK vs MuJoCo engine)
  2. IK solving (analytical + numerical) → move to target pose
  3. Cubic trajectory generation & execution
  4. Multi-waypoint pick-and-place demo

Usage:
    python main.py                  # Run all demos (no viewer)
    python main.py --demo fk        # FK verification only
    python main.py --demo ik        # IK demo
    python main.py --demo traj      # Trajectory demo
    python main.py --demo pick      # Pick-and-place
    python main.py --render         # Enable 3D MuJoCo viewer
"""

import argparse
import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mujoco_sim.kinematics import ForwardKinematics, InverseKinematics
from mujoco_sim.trajectory import TrajectoryGenerator
from mujoco_sim.simulator import MuJoCoSimulator


# ═════════════════════════════════════════════════════════════════════
# Demo 1: Forward Kinematics Verification
# ═════════════════════════════════════════════════════════════════════

def demo_fk(render: bool = False):
    """Compare custom FK output with MuJoCo's internal FK."""
    print("\n" + "=" * 70)
    print("  Demo 1: Forward Kinematics Verification")
    print("=" * 70)

    fk = ForwardKinematics()
    sim = MuJoCoSimulator(render=False)  # headless for FK check

    test_configs = [
        ("Zero pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("Pose A", [0.3, 1.0, -1.2, -0.5, 0.2, 0.1]),
        ("Pose B", [np.pi / 6, np.pi / 2, -np.pi / 3, -np.pi / 4, 0.0, 0.0]),
        ("Pose C", [-0.5, 0.8, -1.8, -1.2, 0.4, -0.2]),
    ]

    print(f"\n{'Config':<12} {'Custom FK pos (mm)':<40} {'MuJoCo EE pos (m)':<40}")
    print("-" * 92)

    for name, angles in test_configs:
        # Custom FK (mm)
        quat, pos_mm = fk.compute(angles)

        # MuJoCo FK — set qpos directly and forward
        sim.reset(angles)
        ee_pos_m = sim.get_ee_position()  # metres
        ee_pos_mm = ee_pos_m * 1000       # convert to mm for comparison

        print(f"{name:<12} [{pos_mm[0]:8.2f}, {pos_mm[1]:8.2f}, {pos_mm[2]:8.2f}]"
              f"          [{ee_pos_mm[0]:8.2f}, {ee_pos_mm[1]:8.2f}, {ee_pos_mm[2]:8.2f}]")

    print("\n  Note: Custom FK and MuJoCo may differ slightly due to frame conventions.")
    print("  The custom FK follows the original SOARM101 DH convention.")
    sim.close()


# ═════════════════════════════════════════════════════════════════════
# Demo 2: Inverse Kinematics
# ═════════════════════════════════════════════════════════════════════

def demo_ik(render: bool = False):
    """Demonstrate analytical and numerical IK, then move the arm."""
    print("\n" + "=" * 70)
    print("  Demo 2: Inverse Kinematics — Analytical vs Numerical")
    print("=" * 70)

    fk = ForwardKinematics()
    ik = InverseKinematics()
    sim = MuJoCoSimulator(render=render)

    # Ground truth angles
    gt_angles = [0.3, 1.2, -1.5, -0.8, 0.2, 0.1]
    quat_target, pos_target = fk.compute(gt_angles)

    print(f"\n  Ground truth angles (deg):  {np.rad2deg(gt_angles)}")
    print(f"  Target position (mm):       {pos_target}")
    print(f"  Target quaternion [xyzw]:   {quat_target}")

    # Analytical IK
    angles_a = ik.analytical(quat_target, pos_target)
    _, pos_a = fk.compute(angles_a)
    err_a = np.linalg.norm(pos_a - pos_target)

    # Numerical IK
    angles_n = ik.numerical(quat_target, pos_target)
    _, pos_n = fk.compute(angles_n)
    err_n = np.linalg.norm(pos_n - pos_target)

    print(f"\n  Analytical IK angles (deg): {np.rad2deg(angles_a)}")
    print(f"    Position error: {err_a:.4f} mm")
    print(f"\n  Numerical IK angles (deg):  {np.rad2deg(angles_n)}")
    print(f"    Position error: {err_n:.4f} mm")

    # Execute in MuJoCo
    if render:
        print("\n  → Moving arm to IK solution in MuJoCo viewer...")
        tg = TrajectoryGenerator()
        current = [0.0] * 6
        traj = tg.joint_space_trajectory(current, angles_n, duration=3.0, frequency=50)
        sim.execute_trajectory(traj, frequency=50)
        time.sleep(1)

    sim.close()


# ═════════════════════════════════════════════════════════════════════
# Demo 3: Cubic Trajectory Generation & Execution
# ═════════════════════════════════════════════════════════════════════

def demo_trajectory(render: bool = False):
    """Generate and execute a cubic trajectory in joint space."""
    print("\n" + "=" * 70)
    print("  Demo 3: Cubic Trajectory Generation & Execution")
    print("=" * 70)

    tg = TrajectoryGenerator()
    sim = MuJoCoSimulator(render=render)

    start_angles = [0.0, 0.8, -1.5, -1.0, 0.0, 0.0]
    end_angles = [0.5, 1.2, -1.0, -0.5, 0.3, 0.1]
    duration = 3.0
    freq = 50.0

    print(f"\n  Start angles (deg): {np.rad2deg(start_angles)}")
    print(f"  End angles (deg):   {np.rad2deg(end_angles)}")
    print(f"  Duration: {duration}s, Frequency: {freq} Hz")

    # Generate trajectory
    traj = tg.joint_space_trajectory(start_angles, end_angles, duration, freq)
    print(f"  Generated {len(traj)} waypoints")

    # Verify smoothness
    vels = np.diff(traj, axis=0) * freq
    accels = np.diff(vels, axis=0) * freq
    print(f"  Max joint velocity:     {np.max(np.abs(vels)):.4f} rad/s")
    print(f"  Max joint acceleration: {np.max(np.abs(accels)):.4f} rad/s²")

    # Set initial pose and execute
    sim.reset(start_angles)
    print(f"\n  Executing trajectory...")
    states = sim.execute_trajectory(traj, frequency=freq, record=True)

    if states:
        final = states[-1]
        print(f"  Final joint positions (deg): {np.rad2deg(final['joint_pos'])}")
        print(f"  Target joint positions (deg): {np.rad2deg(end_angles)}")
        err = np.abs(final['joint_pos'] - np.array(end_angles))
        print(f"  Joint errors (deg): {np.rad2deg(err)}")

    sim.close()


# ═════════════════════════════════════════════════════════════════════
# Demo 4: Multi-Waypoint Pick-and-Place
# ═════════════════════════════════════════════════════════════════════

def demo_pick_and_place(render: bool = False):
    """Multi-waypoint trajectory simulating a pick-and-place operation."""
    print("\n" + "=" * 70)
    print("  Demo 4: Multi-Waypoint Pick-and-Place Simulation")
    print("=" * 70)

    fk = ForwardKinematics()
    tg = TrajectoryGenerator()
    sim = MuJoCoSimulator(render=render)

    # Define waypoints (joint space)
    waypoints = [
        [0.0, 0.8, -1.5, -1.0, 0.0, 0.0],     # Home
        [0.3, 1.0, -1.2, -0.8, 0.0, 0.0],      # Pre-pick approach
        [0.3, 1.3, -1.0, -0.5, 0.0, 0.0],       # Pick position
        [0.3, 1.0, -1.2, -0.8, 0.0, 0.0],       # Lift
        [-0.3, 1.0, -1.2, -0.8, 0.0, 0.0],      # Transport
        [-0.3, 1.3, -1.0, -0.5, 0.0, 0.0],      # Place position
        [-0.3, 1.0, -1.2, -0.8, 0.0, 0.0],      # Retract
        [0.0, 0.8, -1.5, -1.0, 0.0, 0.0],       # Home
    ]
    durations = [2.0, 1.5, 1.5, 1.5, 2.5, 1.5, 1.5, 2.0]  # seconds per segment
    freq = 50.0

    print(f"\n  Waypoints: {len(waypoints)}")
    for i, wp in enumerate(waypoints):
        _, pos = fk.compute(wp)
        label = ["Home", "Pre-pick", "Pick", "Lift", "Transport", "Place", "Retract", "Home"][i]
        print(f"    {i}: {label:<12} → EE at [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] mm")

    # Generate full trajectory
    traj = tg.multi_waypoint_joint_trajectory(waypoints, durations, freq)
    print(f"\n  Total trajectory points: {len(traj)}")
    print(f"  Total duration: {sum(durations):.1f}s")

    # Execute
    sim.reset(waypoints[0])
    print("  Executing pick-and-place trajectory...")
    states = sim.execute_trajectory(traj, frequency=freq, record=True)

    if states:
        print(f"\n  Trajectory completed.")
        final_pos = np.rad2deg(states[-1]['joint_pos'])
        home_pos = np.rad2deg(waypoints[-1])
        print(f"  Final position (deg):  {final_pos}")
        print(f"  Home position (deg):   {home_pos}")
        err = np.abs(final_pos - home_pos)
        print(f"  Return-to-home error (deg): {err}")

    sim.close()


# ═════════════════════════════════════════════════════════════════════
# Demo 5: Task-Space Trajectory with IK
# ═════════════════════════════════════════════════════════════════════

def demo_task_space(render: bool = False):
    """Generate a Cartesian trajectory and convert via IK."""
    print("\n" + "=" * 70)
    print("  Demo 5: Task-Space Trajectory with IK Conversion")
    print("=" * 70)

    fk = ForwardKinematics()
    tg = TrajectoryGenerator()
    sim = MuJoCoSimulator(render=render)

    # Compute start/end poses from known joint configs
    start_angles = [0.0, 0.8, -1.5, -1.0, 0.0, 0.0]
    end_angles = [0.5, 1.2, -1.0, -0.5, 0.3, 0.1]
    q_start, p_start = fk.compute(start_angles)
    q_end, p_end = fk.compute(end_angles)

    print(f"\n  Start EE position (mm): {p_start}")
    print(f"  End EE position (mm):   {p_end}")

    # Generate task-space trajectory
    duration = 4.0
    freq = 50.0
    task_traj = tg.task_space_trajectory(
        start_pos=p_start.tolist(),
        end_pos=p_end.tolist(),
        start_euler=[0, 0, 0],
        end_euler=[0.3, 0.2, 0.1],
        duration=duration,
        frequency=freq,
    )
    print(f"  Task-space waypoints: {len(task_traj)}")

    # Convert to joint space
    print("  Converting to joint space via numerical IK...")
    joint_traj = tg.task_to_joint_trajectory(
        task_traj, method="numerical", initial_angles=start_angles
    )
    print(f"  Joint-space waypoints: {len(joint_traj)}")

    # Execute
    sim.reset(start_angles)
    print("  Executing task-space trajectory...")
    sim.execute_trajectory(joint_traj, frequency=freq)

    sim.close()


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

DEMOS = {
    "fk": demo_fk,
    "ik": demo_ik,
    "traj": demo_trajectory,
    "pick": demo_pick_and_place,
    "task": demo_task_space,
}


def main():
    parser = argparse.ArgumentParser(
        description="SO-ARM100 MuJoCo Simulation Demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available demos:
  fk    — Forward Kinematics verification (custom FK vs MuJoCo)
  ik    — Inverse Kinematics (analytical & numerical)
  traj  — Cubic trajectory generation & execution
  pick  — Multi-waypoint pick-and-place simulation
  task  — Task-space trajectory with IK conversion
        """,
    )
    parser.add_argument(
        "--demo",
        choices=list(DEMOS.keys()),
        default=None,
        help="Run a specific demo. Omit to run all.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable MuJoCo 3D viewer (requires display).",
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     SO-ARM100  MuJoCo  Model-Based  Manipulation Demo      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if args.demo:
        DEMOS[args.demo](render=args.render)
    else:
        for name, func in DEMOS.items():
            func(render=args.render)

    print("\n✅ All demos completed.\n")


if __name__ == "__main__":
    main()
