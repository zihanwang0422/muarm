"""Standalone trajectory demo: Cartesian waypoints -> IK -> smooth joint trajectory."""
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

# Define Cartesian waypoints (x, y, z).
WAYPOINTS = [
    np.array([0.48, -0.15, 0.30]),
    np.array([0.55, 0.00, 0.28]),
    np.array([0.42, 0.16, 0.33]),
]


class TrajectoryRunner:
    def __init__(self, scene_xml, panda_xml):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.kin = PandaKinematics(ee_frame="hand")
        self.kin.build_from_mjcf(panda_xml)
        self.traj = TrajectoryGenerator()
        self._path_cache = []

    @staticmethod
    def _target_tf(xyz):
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = np.diag([1.0, -1.0, -1.0])
        tf[:3, 3] = xyz
        return tf

    def _build_joint_waypoints(self, cartesian_points, q_home):
        joint_waypoints = [q_home.copy()]
        for p in cartesian_points:
            q_sol, info = self.kin.ik(self._target_tf(p), q_init=joint_waypoints[-1])
            print(
                f"Trajectory waypoint={p.tolist()} success={info['success']} "
                f"err={info['error_norm']:.5f}"
            )
            joint_waypoints.append(q_sol[:7].copy())
        return joint_waypoints

    def _draw_overlay(self, viewer, wp_positions):
        ee = self.data.body("hand").xpos.copy()
        if not self._path_cache or np.linalg.norm(ee - self._path_cache[-1]) > 0.002:
            self._path_cache.append(ee)
        if len(self._path_cache) > 400:
            self._path_cache.pop(0)

        ngeom = 0
        viewer.user_scn.ngeom = 0

        for wp in wp_positions:
            if ngeom >= viewer.user_scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.02, 0.0, 0.0]),
                pos=np.array(wp, dtype=np.float64),
                mat=np.eye(3).ravel(),
                rgba=np.array([1.0, 0.7, 0.2, 0.9], dtype=np.float32),
            )
            ngeom += 1

        for i in range(len(self._path_cache) - 1):
            if ngeom >= viewer.user_scn.maxgeom:
                break
            g = viewer.user_scn.geoms[ngeom]
            mujoco.mjv_connector(
                g,
                mujoco.mjtGeom.mjGEOM_LINE,
                2.5,
                self._path_cache[i],
                self._path_cache[i + 1],
            )
            g.rgba[:] = np.array([0.1, 0.9, 0.2, 0.9], dtype=np.float32)
            ngeom += 1

        viewer.user_scn.ngeom = ngeom

    def run(self, waypoints):
        home_q = self.model.key_qpos[0].copy()
        self.data.qpos[:] = home_q
        self.data.ctrl[:] = self.model.key_ctrl[0].copy()

        q_waypoints = self._build_joint_waypoints(waypoints, home_q[:7])
        wp_positions = [self.kin.get_ee_position(q) for q in q_waypoints[1:]]
        traj = self.traj.multi_waypoint(q_waypoints, segment_duration=1.2, dt=self.model.opt.timestep)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 2.25
            viewer.cam.azimuth = -50
            viewer.cam.elevation = -25

            while viewer.is_running():
                self._path_cache = []
                for q in traj["positions"]:
                    if not viewer.is_running():
                        return
                    self.data.ctrl[:7] = q
                    self.data.ctrl[7] = 255.0
                    mujoco.mj_step(self.model, self.data)
                    self._draw_overlay(viewer, wp_positions)
                    viewer.sync()
                    time.sleep(self.model.opt.timestep)


if __name__ == "__main__":
    runner = TrajectoryRunner(SCENE_XML, PANDA_XML)
    runner.run(WAYPOINTS)
