"""Standalone FK demo: define joint targets in this script and run to visualize."""
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematic.panda_kinematics import PandaKinematics
from kinematic.trajectory import TrajectoryGenerator


SCENE_XML = str(ROOT / "models" / "franka_emika_panda" / "scene.xml")
PANDA_XML = str(ROOT / "models" / "franka_emika_panda" / "panda.xml")

# Define joint-space FK targets (rad).
FK_TARGETS = [
    np.array([0.0, -0.3, 0.0, -1.7, 0.0, 1.6, -0.7]),
    np.array([0.4, -0.5, 0.2, -1.8, 0.2, 1.7, -0.2]),
    np.array([-0.4, 0.2, -0.2, -2.0, -0.2, 1.9, -1.0]),
]


class FKRunner:
    def __init__(self, scene_xml, panda_xml):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.kin = PandaKinematics(ee_frame="hand")
        self.kin.build_from_mjcf(panda_xml)
        self.traj = TrajectoryGenerator()

    def _draw_sphere(self, viewer, pos, radius=0.025, rgba=None):
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0]),
            pos=np.array(pos, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array(rgba or [0.2, 0.4, 1.0, 0.85], dtype=np.float32),
        )

    def run(self, targets):
        home_q = self.model.key_qpos[0].copy()
        self.data.qpos[:] = home_q
        self.data.ctrl[:] = self.model.key_ctrl[0].copy()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 2.3
            viewer.cam.azimuth = -40
            viewer.cam.elevation = -22

            current_q = self.data.qpos[:7].copy()
            while viewer.is_running():
                for q_target in targets:
                    segment = self.traj.cubic(
                        current_q,
                        q_target,
                        duration=2.0,
                        dt=self.model.opt.timestep,
                    )
                    ee_target = self.kin.get_ee_position(q_target)

                    for q in segment["positions"]:
                        if not viewer.is_running():
                            return
                        self.data.ctrl[:7] = q
                        self.data.ctrl[7] = 255.0
                        mujoco.mj_step(self.model, self.data)
                        self._draw_sphere(viewer, ee_target)
                        viewer.sync()
                        time.sleep(self.model.opt.timestep)

                    current_q = q_target.copy()


if __name__ == "__main__":
    runner = FKRunner(SCENE_XML, PANDA_XML)
    runner.run(FK_TARGETS)
