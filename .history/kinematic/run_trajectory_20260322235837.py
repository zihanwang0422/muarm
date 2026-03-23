"""Standalone figure-8 trajectory demo.

The robot's current EE position (home keyframe) is used as the crossing point
of the figure-8.  A Catmull-Rom spline is built through IK-solved waypoints,
giving continuous, pause-free motion.  The 8-shape loops indefinitely.

Figure-8 geometry (horizontal ∞):
    x  = center_x        (constant depth)
    y  = center_y + Ry * sin(t)        (side-to-side)
    z  = center_z + Rz * sin(2t) / 2  (up-down at 2x freq)

Tune RADIUS_Y / RADIUS_Z / N_WP / SEG_DUR at the top of the file.
"""
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

# ── Figure-8 parameters (tune here) ──────────────────────────────────────────
N_WP      = 16     # IK waypoints per loop  (more → smoother, slower to build)
RADIUS_Y  = 0.30   # half-width  in Y direction (m)
RADIUS_Z  = 0.30   # half-height in Z direction (m)
SEG_DUR   = 0.50   # seconds per Catmull-Rom segment → N_WP * SEG_DUR = loop time
# ─────────────────────────────────────────────────────────────────────────────


class TrajectoryRunner:
    def __init__(self, scene_xml, panda_xml):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data  = mujoco.MjData(self.model)
        self.kin   = PandaKinematics(ee_frame="hand")
        self.kin.build_from_mjcf(panda_xml)
        self.traj  = TrajectoryGenerator()
        self._trail = []   # live EE position trail

    @staticmethod
    def _target_tf(xyz):
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = np.diag([1.0, -1.0, -1.0])
        tf[:3, 3] = xyz
        return tf

    # ── Geometry ──────────────────────────────────────────────────────────────

    def _figure8_cartesian(self, center, n=N_WP):
        """Sample *n* Cartesian points on a horizontal figure-8 centred at
        *center*.  t=0  (and t=π) land exactly on the crossing point."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = []
        for ti in t:
            y = center[1] + RADIUS_Y * np.sin(ti)
            z = center[2] + RADIUS_Z * np.sin(2 * ti) / 2.0
            pts.append(np.array([center[0], y, z]))
        return pts   # length-n list of (3,) arrays

    def _build_joint_waypoints(self, cartesian_pts, q_seed):
        """IK for every Cartesian waypoint, warmstarted from the previous
        solution for stability.  Returns (N, 7) ndarray."""
        q_cur = q_seed.copy()
        q_list = []
        for i, p in enumerate(cartesian_pts):
            q_sol, info = self.kin.ik(
                self._target_tf(p), q_init=q_cur, max_iters=200
            )
            ok = "✓" if info["success"] else "✗"
            print(f"  [{i+1:02d}/{len(cartesian_pts)}] {ok}  "
                  f"err={info['error_norm']:.5f}  pos={np.round(p, 3).tolist()}")
            q_list.append(q_sol[:7].copy())
            q_cur = q_sol[:7].copy()
        return np.array(q_list)   # (N, 7)

    # ── Overlay drawing ───────────────────────────────────────────────────────

    def _draw_overlay(self, viewer):
        """Draw the live EE trail (cyan)."""
        ee = self.data.body("hand").xpos.copy()
        if not self._trail or np.linalg.norm(ee - self._trail[-1]) > 0.001:
            self._trail.append(ee.copy())
        # Keep exactly one full figure-8 loop in the buffer
        if len(self._trail) > self._trail_max:
            self._trail.pop(0)

        viewer.user_scn.ngeom = 0
        ngeom = 0
        for i in range(len(self._trail) - 1):
            if ngeom >= viewer.user_scn.maxgeom:
                break
            g = viewer.user_scn.geoms[ngeom]
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_LINE, 3.0,
                np.array(self._trail[i],     dtype=np.float64),
                np.array(self._trail[i + 1], dtype=np.float64),
            )
            g.rgba[:] = np.array([0.1, 0.95, 0.4, 0.95], dtype=np.float32)
            ngeom += 1
        viewer.user_scn.ngeom = ngeom

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self):
        # Reset to home keyframe
        home_q = self.model.key_qpos[0].copy()
        self.data.qpos[:] = home_q
        self.data.ctrl[:] = self.model.key_ctrl[0].copy()
        mujoco.mj_forward(self.model, self.data)

        # Figure-8 centre = current EE position
        center = self.data.body("hand").xpos.copy()
        print(f"Figure-8 centre: {np.round(center, 4).tolist()}")
        print(f"Ry={RADIUS_Y} m   Rz={RADIUS_Z} m   "
              f"N_WP={N_WP}   SEG_DUR={SEG_DUR}s  "
              f"→ loop≈{N_WP * SEG_DUR:.1f}s")

        # Build Cartesian waypoints and solve IK
        fig8_xyz = self._figure8_cartesian(center)
        print(f"Solving IK for {len(fig8_xyz)} waypoints …")
        q_wps = self._build_joint_waypoints(fig8_xyz, home_q[:7])

        # Smooth closed-loop Catmull-Rom trajectory
        traj = self.traj.catmull_rom(
            q_wps, segment_duration=SEG_DUR,
            dt=self.model.opt.timestep, closed=True
        )
        positions = traj["positions"]
        # Trail buffer: hold one full loop + small margin so the complete
        # figure-8 is always visible without old segments disappearing mid-loop.
        self._trail_max = len(positions) + 200
        print(f"Trajectory ready: {len(positions)} steps "
              f"(loop ≈ {len(positions) * self.model.opt.timestep:.1f}s)")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 2.2
            viewer.cam.azimuth  = -40
            viewer.cam.elevation = -24

            while viewer.is_running():
                for q in positions:
                    if not viewer.is_running():
                        return
                    self.data.ctrl[:7] = q
                    self.data.ctrl[7]  = 255.0
                    mujoco.mj_step(self.model, self.data)
                    self._draw_overlay(viewer)
                    viewer.sync()
                    time.sleep(self.model.opt.timestep)


if __name__ == "__main__":
    runner = TrajectoryRunner(SCENE_XML, PANDA_XML)
    runner.run()
