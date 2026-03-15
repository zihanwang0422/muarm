"""Example: Panda joint impedance control in MuJoCo simulation.

Demonstrates:
- Using the MuJoCoViewer base class
- Joint-space impedance controller
- Tracking a desired joint configuration
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src import MuJoCoViewer
from control import ImpedanceController

SCENE_XML = "models/franka_emika_panda/scene.xml"


class ImpedanceDemo(MuJoCoViewer):
    def __init__(self):
        super().__init__(SCENE_XML, distance=3, azimuth=-45, elevation=-30)

    def runBefore(self):
        # Set initial position from keyframe
        if self.model.nkey > 0:
            home = self.model.key_qpos[0]
            for i in range(self.model.nq):
                self.data.qpos[i] = home[i]

        # Target: move joint1 and joint4
        self.q_desired = np.array([0.5, -0.5, 0.0, -1.5, 0.0, 1.5, 0.5])

        # Impedance controller
        self.controller = ImpedanceController(
            n_joints=7,
            kp=[600, 600, 600, 600, 250, 150, 50],
            kd=[50, 50, 50, 50, 20, 20, 10],
            mode='joint',
        )
        self.step_count = 0

    def runFunc(self):
        q = self.data.qpos[:7].copy()
        qdot = self.data.qvel[:7].copy()

        tau = self.controller.compute_joint(self.q_desired, q, qdot)
        self.data.ctrl[:7] = tau

        self.step_count += 1
        if self.step_count % 500 == 0:
            error = np.linalg.norm(self.q_desired - q)
            print(f"Step {self.step_count}: joint error = {error:.4f}")


if __name__ == "__main__":
    demo = ImpedanceDemo()
    demo.run_loop()
