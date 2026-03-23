"""Standalone IK demo: define Cartesian goals and visualize motion directly."""
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

# Define Cartesian IK goals (x, y, z).
IK_GOALS = [
    np.array([0.45, 0.10, 0.35]),
    np.array([0.55, -0.05, 0.30]),
    np.array([0.50, 0.15, 0.25]),
]


class IKRunner:
    def __init__(self, scene_xml, panda_xml):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.kin = PandaKinematics(ee_frame="hand")
        self.kin.build_from_mjcf(panda_xml)
        self.traj = TrajectoryGenerator()
        self._trail = []

    @staticmethod
    def _target_tf(xyz):
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = np.diag([1.0, -1.0, -1.0])
        tf[:3, 3] = xyz
        return tf

    def _draw_overlay(self, viewer, target):
        ee = self.data.body("hand").xpos.copy()
        if not self._trail or np.linalg.norm(ee - self._trail[-1]) > 0.003:
            self._trail.append(ee)
        if len(self._trail) > 200:
            self._trail.pop(0)

        ngeom = 0
        viewer.user_scn.ngeom = 0

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.028, 0.0, 0.0]),
            pos=np.array(target, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array([1.0, 0.15, 0.15, 0.85], dtype=np.float32),
        )
        ngeom += 1

        for i in range(len(self._trail) - 1):
            if ngeom >= viewer.user_scn.maxgeom:
                break
            g = viewer.user_scn.geoms[ngeom]
            mujoco.mjv_connector(
                g,
                mujoco.mjtGeom.mjGEOM_LINE,
                2.0,
                self._trail[i],
                self._trail[i + 1],
            )
            g.rgba[:] = np.array([0.2, 0.8, 1.0, 0.9], dtype=np.float32)
            ngeom += 1

        viewer.user_scn.ngeom = ngeom

    def run(self, goals):
        home_q = self.model.key_qpos[0].copy()
        self.data.qpos[:] = home_q
        self.data.ctrl[:] = self.model.key_ctrl[0].copy()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 2.2
            viewer.cam.azimuth = -35
            viewer.cam.elevation = -24

            q_seed = self.data.qpos[:7].copy()
            while viewer.is_running():
                for goal in goals:
                    tf = self._target_tf(goal)
                    q_sol, info = self.kin.ik(tf, q_init=q_seed, max_iters=250)
                    print(
                        f"IK goal={goal.tolist()} success={info['success']} "
                        f"iters={info['iterations']} err={info['error_norm']:.5f}"
                    )

                    segment = self.traj.cubic(
                        q_seed,
                        q_sol[:7],
                        duration=2.0,
                        dt=self.model.opt.timestep,
                    )

                    for q in segment["positions"]:
                        if not viewer.is_running():
                            return
                        self.data.ctrl[:7] = q
                        self.data.ctrl[7] = 255.0
                        mujoco.mj_step(self.model, self.data)
                        self._draw_overlay(viewer, goal)
                        viewer.sync()
                        time.sleep(self.model.opt.timestep)

                    q_seed = q_sol[:7].copy()
                    for _ in range(80):
                        if not viewer.is_running():
                            return
                        mujoco.mj_step(self.model, self.data)
                        self._draw_overlay(viewer, goal)
                        viewer.sync()
                        time.sleep(self.model.opt.timestep)


if __name__ == "__main__":
    runner = IKRunner(SCENE_XML, PANDA_XML)
    runner.run(IK_GOALS)
