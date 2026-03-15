#!/usr/bin/env python3
"""
SO-ARM100 MuJoCo Simulation — Main Entry Point

Demos:
  fk     FK verification (custom FK vs MuJoCo engine)
  ik     Numerical IK solve → move arm to target
  traj   Cubic joint-space trajectory
  pick   Multi-waypoint pick-and-place
  task   Task-space trajectory with IK conversion
  view   Just open the interactive viewer

Usage:
    python -m mujoco_sim.main                  # all demos (headless)
    python -m mujoco_sim.main --demo fk        # one demo
    python -m mujoco_sim.main --render         # with 3D viewer
    python -m mujoco_sim.main --demo view      # interactive viewer only
"""

import argparse, sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mujoco_sim.kinematics import ForwardKinematics, InverseKinematics
from mujoco_sim.trajectory import TrajectoryGenerator
from mujoco_sim.simulator import MuJoCoSimulator


# ═════════════════════════════════════════════════════════════════
def demo_fk(render=False):
    """Compare custom FK with MuJoCo's internal FK."""
    print("\n" + "=" * 70)
    print("  Demo 1 · Forward Kinematics Verification")
    print("=" * 70)

    fk = ForwardKinematics()
    sim = MuJoCoSimulator(render=False)

    configs = [
        ("Zero",  [0, 0, 0, 0, 0]),
        ("Home",  [0, -1.57, 1.57, 1.57, -1.57]),
        ("TestA", [0.3, -1.0, 1.2, 0.5, -0.2]),
        ("TestB", [-0.5, -2.0, 2.5, 1.0, 0.8]),
    ]

    fmt = "{:<8} {:>32}  {:>32}  {:>10}"
    print(fmt.format("Name", "Custom FK pos (m)", "MuJoCo EE pos (m)", "Δ (mm)"))
    print("-" * 86)

    for name, angles in configs:
        p_fk, _ = fk.compute(angles)
        sim.reset(angles)
        p_mj = sim.get_ee_position()
        err = np.linalg.norm(p_fk - p_mj) * 1000
        pf = lambda v: f"[{v[0]:8.5f}, {v[1]:8.5f}, {v[2]:8.5f}]"
        print(fmt.format(name, pf(p_fk), pf(p_mj), f"{err:.3f}"))

    sim.close()


# ═════════════════════════════════════════════════════════════════
def demo_ik(render=False):
    """Numerical IK round-trip test, then move arm."""
    print("\n" + "=" * 70)
    print("  Demo 2 · Inverse Kinematics (Numerical)")
    print("=" * 70)

    fk = ForwardKinematics()
    ik = InverseKinematics()

    gt = [0.3, -1.0, 1.2, 0.5, -0.2]
    p_target, q_target = fk.compute(gt)
    print(f"\n  GT angles (deg):  {np.rad2deg(gt)}")
    print(f"  Target pos (m):   {p_target}")
    print(f"  Target quat:      {q_target}")

    home = [0, -1.57, 1.57, 1.57, -1.57]
    recovered = ik.numerical(p_target, q_target, initial_angles=home)
    p_rec, _ = fk.compute(recovered)
    err = np.linalg.norm(p_rec - p_target) * 1000
    print(f"\n  IK angles (deg):  {np.rad2deg(recovered)}")
    print(f"  Pos error:        {err:.4f} mm")

    if render:
        sim = MuJoCoSimulator(render=True)
        sim.reset_to_keyframe("home")
        tg = TrajectoryGenerator()
        home = [0, -1.57, 1.57, 1.57, -1.57]
        traj = tg.joint_trajectory(home, recovered, duration=3.0, frequency=50)
        print("  → Moving to IK solution …")
        sim.execute_trajectory(traj, frequency=50)
        time.sleep(1.5)
        sim.close()


# ═════════════════════════════════════════════════════════════════
def demo_traj(render=False):
    """Generate and execute cubic joint trajectory."""
    print("\n" + "=" * 70)
    print("  Demo 3 · Cubic Joint-Space Trajectory")
    print("=" * 70)

    tg = TrajectoryGenerator()
    sim = MuJoCoSimulator(render=render)

    start = [0, -1.57, 1.57, 1.57, -1.57]  # home
    end = [0.5, -1.0, 1.2, 0.8, -0.5]
    dur, freq = 3.0, 50.0

    traj = tg.joint_trajectory(start, end, dur, freq)
    print(f"\n  {len(traj)} waypoints, {dur}s, {freq}Hz")

    vels = np.diff(traj, axis=0) * freq
    print(f"  Max velocity:     {np.max(np.abs(vels)):.4f} rad/s")

    sim.reset(start)
    states = sim.execute_trajectory(traj, frequency=freq, record=True)
    if states:
        final = states[-1]["arm_pos"]
        err = np.abs(final - np.array(end))
        print(f"  Final error (deg): {np.rad2deg(err)}")
    if render:
        time.sleep(1)
    sim.close()


# ═════════════════════════════════════════════════════════════════
def demo_pick(render=False):
    """Multi-waypoint pick-and-place."""
    print("\n" + "=" * 70)
    print("  Demo 4 · Multi-Waypoint Pick-and-Place")
    print("=" * 70)

    fk = ForwardKinematics()
    tg = TrajectoryGenerator()
    sim = MuJoCoSimulator(render=render)

    waypoints = [
        [0, -1.57, 1.57, 1.57, -1.57],       # Home
        [0.3, -1.2, 1.3, 1.2, -1.0],          # Pre-pick
        [0.3, -0.8, 1.0, 0.8, -1.0],          # Pick
        [0.3, -1.2, 1.3, 1.2, -1.0],          # Lift
        [-0.3, -1.2, 1.3, 1.2, -1.0],         # Transport
        [-0.3, -0.8, 1.0, 0.8, -1.0],         # Place
        [-0.3, -1.2, 1.3, 1.2, -1.0],         # Retract
        [0, -1.57, 1.57, 1.57, -1.57],        # Home
    ]
    durs = [2.0, 1.5, 1.5, 1.5, 2.5, 1.5, 2.0]
    labels = ["Home", "Pre-pick", "Pick", "Lift", "Transport", "Place", "Retract", "Home"]
    freq = 50.0

    for i, wp in enumerate(waypoints):
        p, _ = fk.compute(wp)
        print(f"  {labels[i]:<12} EE → [{p[0]*1000:.1f}, {p[1]*1000:.1f}, {p[2]*1000:.1f}] mm")

    traj = tg.multi_waypoint(waypoints, durs, freq)
    print(f"\n  Total: {len(traj)} pts, {sum(durs):.1f}s")

    sim.reset(waypoints[0])
    sim.execute_trajectory(traj, frequency=freq)
    if render:
        time.sleep(1)
    sim.close()


# ═════════════════════════════════════════════════════════════════
def demo_task(render=False):
    """Task-space trajectory with IK conversion."""
    print("\n" + "=" * 70)
    print("  Demo 5 · Task-Space Trajectory → IK")
    print("=" * 70)

    fk = ForwardKinematics()
    tg = TrajectoryGenerator()
    sim = MuJoCoSimulator(render=render)

    start_a = [0, -1.57, 1.57, 1.57, -1.57]
    end_a = [0.5, -1.0, 1.2, 0.8, -0.5]
    p_s, _ = fk.compute(start_a)
    p_e, _ = fk.compute(end_a)
    print(f"\n  Start EE (m): {p_s}")
    print(f"  End   EE (m): {p_e}")

    task_traj = tg.task_trajectory(
        p_s.tolist(), p_e.tolist(),
        [0, 0, 0], [0.3, 0.2, 0.1],
        duration=4.0, frequency=50,
    )
    print(f"  Task-space pts: {len(task_traj)}")

    print("  Converting via IK …")
    joint_traj = tg.task_to_joint(task_traj, initial_angles=start_a)
    print(f"  Joint-space pts: {len(joint_traj)}")

    sim.reset(start_a)
    sim.execute_trajectory(joint_traj, frequency=50)
    if render:
        time.sleep(1)
    sim.close()


# ═════════════════════════════════════════════════════════════════
def demo_view(render=True):
    """Open interactive viewer with home pose."""
    print("\n  Opening interactive MuJoCo viewer (close window to exit) …")
    sim = MuJoCoSimulator(render=False)
    sim.reset_to_keyframe("home")
    sim.launch_viewer()


# ═════════════════════════════════════════════════════════════════
DEMOS = {
    "fk": demo_fk, "ik": demo_ik, "traj": demo_traj,
    "pick": demo_pick, "task": demo_task, "view": demo_view,
}


def main():
    parser = argparse.ArgumentParser(description="SO-ARM100 MuJoCo Demos")
    parser.add_argument("--demo", choices=list(DEMOS), default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       SO-ARM100  MuJoCo  Model-Based  Manipulation         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if args.demo:
        DEMOS[args.demo](render=args.render or args.demo == "view")
    else:
        for name in ["fk", "ik", "traj", "pick", "task"]:
            DEMOS[name](render=args.render)

    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
