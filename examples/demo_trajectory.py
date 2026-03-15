"""Cubic trajectory execution — live MuJoCo visualization.

Generates a multi-waypoint cubic trajectory along a figure-8 (lemniscate)
path and plays it back inside MuJoCo.  Coloured spheres mark each waypoint;
a green sphere tracks the live end-effector; a red line traces the EE path.

Run:
    python examples/demo_trajectory.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from src.kinematics_vis import KinematicsVisualizer

SCENE = "models/franka_emika_panda/scene.xml"
PANDA = "models/franka_emika_panda/panda.xml"

# Figure-8 (lemniscate) in the XY plane, z = 0.42
# Parametric: x = cx + Rx*sin(t),  y = cy + Ry*sin(2t)/2
cx, cy, cz = 0.50, 0.00, 0.42
Rx, Ry = 0.10, 0.14
t_pts = np.linspace(0, 2 * np.pi, 9)   # 9 pts — last closes the loop
waypoints = [
    [round(cx + Rx * np.sin(t), 3),
     round(cy + Ry * np.sin(2 * t) / 2, 3),
     cz]
    for t in t_pts
]

print("Figure-8 trajectory — loops continuously. Close the viewer to exit.")
KinematicsVisualizer(
    scene_xml=SCENE, panda_xml=PANDA,
    mode="trajectory",
    targets=waypoints,
    hold_steps=0,
    distance=2.5, azimuth=-30, elevation=-20,
).run_loop()
