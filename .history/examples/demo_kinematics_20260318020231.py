"""FK / IK / Trajectory — live MuJoCo visualization.

Three stages run in sequence (close the window to advance):
  Stage 1 — FK sweep through 4 joint configurations (blue sphere = EE)
  Stage 2 — IK: robot moves to 4 Cartesian targets (red sphere = target)
  Stage 3 — Trajectory: figure-8 path; coloured spheres = waypoints, red line = EE trail

Run:
    python examples/demo_kinematics.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from src.kinematics_vis import KinematicsVisualizer

SCENE = "models/franka_emika_panda/scene_pos.xml"
PANDA = "models/franka_emika_panda/panda_general.xml"

# ── Stage 1 : FK ──────────────────────────────────────────────────────────────
print("\n[Stage 1] FK demo  — close the viewer window to continue")
KinematicsVisualizer(
    scene_xml=SCENE, panda_xml=PANDA,
    mode="fk",
    targets=[
        [0,    0,    0,   -1.57, 0,    1.57,  -0.79],
        [0.5,  0.3, -0.3, -2.0,  0.2,  2.4,  -0.5 ],
        [-0.5, 0.3,  0.3, -1.8, -0.2,  2.2,   0.5 ],
        [0,   -0.3,  0,   -2.5,  0,    2.2,   0.79],
    ],
    steps_per_target=600, hold_steps=150,
).run_loop()

# ── Stage 2 : IK ──────────────────────────────────────────────────────────────
print("\n[Stage 2] IK demo  — close the viewer window to continue")
KinematicsVisualizer(
    scene_xml=SCENE, panda_xml=PANDA,
    mode="ik",
    targets=[
        [0.50,  0.00, 0.30],
        [0.45,  0.25, 0.45],
        [0.40, -0.25, 0.45],
        [0.35,  0.00, 0.60],
    ],
    steps_per_target=700, hold_steps=200,
).run_loop()

# ── Stage 3 : Trajectory (figure-8) ──────────────────────────────────────────
print("\n[Stage 3] Trajectory demo — close the viewer window to exit")
cx, cy, cz = 0.50, 0.00, 0.42
Rx, Ry = 0.10, 0.14
t_pts = np.linspace(0, 2 * np.pi, 9)
fig8 = [
    [round(cx + Rx * np.sin(t), 3),
     round(cy + Ry * np.sin(2 * t) / 2, 3),
     cz]
    for t in t_pts
]
KinematicsVisualizer(
    scene_xml=SCENE, panda_xml=PANDA,
    mode="trajectory",
    targets=fig8,
    hold_steps=0,
).run_loop()

