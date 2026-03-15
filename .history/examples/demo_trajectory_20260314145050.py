"""Example: Cubic trajectory generation and execution in MuJoCo.

Demonstrates:
- Multi-waypoint trajectory planning
- Executing joint trajectories on the Panda robot
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src import MuJoCoViewer
from kinematic import TrajectoryGenerator

SCENE_XML = "models/franka_emika_panda/scene.xml"


class TrajectoryDemo(MuJoCoViewer):
    def __init__(self):
        super().__init__(SCENE_XML, distance=3, azimuth=-45, elevation=-30)

    def runBefore(self):
        # Home position
        if self.model.nkey > 0:
            home = self.model.key_qpos[0][:7]
        else:
            home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])

        # Define waypoints
        wp1 = home.copy()
        wp2 = home.copy()
        wp2[0] = 0.8  # Rotate joint 1
        wp2[3] = -1.2
        wp3 = home.copy()
        wp3[0] = -0.5
        wp3[1] = 0.3

        # Generate multi-waypoint trajectory
        traj_gen = TrajectoryGenerator()
        self.trajectory = traj_gen.multi_waypoint(
            waypoints=[wp1, wp2, wp3, wp1],
            segment_duration=3.0,
            dt=self.model.opt.timestep,
        )

        self.traj_idx = 0
        print(f"Trajectory: {len(self.trajectory['positions'])} points, "
              f"duration: {self.trajectory['times'][-1]:.1f}s")

    def runFunc(self):
        if self.traj_idx < len(self.trajectory['positions']):
            q_des = self.trajectory['positions'][self.traj_idx]
            self.data.ctrl[:7] = q_des
            self.traj_idx += 1

            if self.traj_idx % 1000 == 0:
                t = self.trajectory['times'][self.traj_idx - 1]
                print(f"  t={t:.2f}s, q={np.round(q_des, 2)}")
        else:
            # Hold last position
            pass


if __name__ == "__main__":
    demo = TrajectoryDemo()
    demo.run_loop()
