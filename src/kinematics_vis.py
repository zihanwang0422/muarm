"""MuJoCo-integrated FK / IK / Trajectory visualizer for Panda robot.

This is the central interface script.  Import ``KinematicsVisualizer`` from
other scripts to get a live MuJoCo viewer that animates kinematics solving.

Usage::

    from src.kinematics_vis import KinematicsVisualizer
    import numpy as np

    vis = KinematicsVisualizer(
        scene_xml  = "models/franka_emika_panda/scene.xml",
        panda_xml  = "models/franka_emika_panda/panda.xml",
        mode       = "ik",          # "fk" | "ik" | "trajectory"
        targets    = [...],         # see each mode below
    )
    vis.run_loop()

Modes
-----
"fk"
    *targets* = list of joint-angle arrays (each length 7).
    Robot animates through each configuration in sequence.

"ik"
    *targets* = list of (3,) Cartesian positions.
    A red sphere marks each target; the robot solves IK and moves there.

"trajectory"
    *targets* = list of (3,) or (7,) waypoints (Cartesian or joint space).
    A cubic trajectory is generated and the robot follows it; coloured
    spheres mark each waypoint.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.mujoco_viewer import MuJoCoViewer
from kinematic.panda_kinematics import PandaKinematics
from kinematic.trajectory import TrajectoryGenerator
from utils.transform import transform2mat

_WAYPOINT_COLORS = [
    [1.0, 0.2, 0.2, 0.9],  # red
    [0.2, 0.8, 0.2, 0.9],  # green
    [0.2, 0.4, 1.0, 0.9],  # blue
    [1.0, 0.8, 0.0, 0.9],  # yellow
    [0.8, 0.2, 0.8, 0.9],  # purple
    [0.0, 0.9, 0.9, 0.9],  # cyan
]

_TRAIL_MAX = 500   # max EE trail points kept in memory


class KinematicsVisualizer(MuJoCoViewer):
    """Live MuJoCo visualizer for FK, IK, and trajectory demos.

    Parameters
    ----------
    scene_xml : str
        Path to the MuJoCo scene XML.
    panda_xml : str
        Path to the Panda robot MJCF (used by Pinocchio for IK).
    mode : str
        "fk", "ik", or "trajectory".
    targets : list
        Mode-dependent target list (see module docstring).
    ee_body : str
        MuJoCo body name for the end-effector (default "hand").
    steps_per_target : int
        Simulation steps allocated for each move (default 800).
    hold_steps : int
        Steps to hold still between moves (default 200).
    """

    def __init__(self, scene_xml, panda_xml=None,
                 mode="ik", targets=None, ee_body="hand",
                 steps_per_target=800, hold_steps=200,
                 distance=3, azimuth=-45, elevation=-30):
        super().__init__(scene_xml, distance, azimuth, elevation)
        self.mode           = mode
        self.targets        = targets or []
        self.ee_body        = ee_body
        self.steps_per_move = steps_per_target
        self.hold_steps     = hold_steps

        panda_xml = panda_xml or os.path.join(
            os.path.dirname(scene_xml), "panda.xml")
        self.kin = PandaKinematics(ee_frame=ee_body)
        self.kin.build_from_mjcf(panda_xml)
        self.tg  = TrajectoryGenerator()

        # internal state
        self._phase      = "init"   # init | move | hold
        self._traj       = None     # (T, 7) joint trajectory
        self._traj_idx   = 0        # current position in trajectory
        self._target_idx = 0        # index into self.targets
        self._hold_cnt   = 0
        self._ik_sols    = []       # pre-computed IK solutions
        self._ee_trail   = []       # EE position trail (trajectory mode)

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def runBefore(self):
        # Apply home keyframe
        home_q = self.model.key_qpos[0].copy()
        self.data.qpos[:] = home_q
        self.data.ctrl[:] = self.model.key_ctrl[0].copy()

        # Pre-compute IK for every target
        if self.mode == "ik":
            self._precompute_ik(home_q[:7])
        elif self.mode == "trajectory":
            self._precompute_trajectory(home_q[:7])
        # For "fk", self.targets already are joint configs

        self._phase = "hold"
        self._hold_cnt = self.hold_steps   # brief pause before first move

    def runFunc(self):
        # --- clear overlay each frame ---
        self.handle.user_scn.ngeom = 0

        # --- record EE trail in trajectory mode ---
        if self.mode == "trajectory" and self._phase == "move":
            ee_pos = self.get_body_position(self.ee_body)
            if (not self._ee_trail or
                    np.linalg.norm(ee_pos - self._ee_trail[-1]) > 0.003):
                self._ee_trail.append(ee_pos.copy())
                if len(self._ee_trail) > _TRAIL_MAX:
                    self._ee_trail.pop(0)

        # --- draw visual markers ---
        self._draw_markers()

        # --- state machine ---
        if self._phase == "hold":
            self._hold_cnt -= 1
            if self._hold_cnt <= 0:
                # advance to next target
                if self._target_idx >= len(self.targets):
                    self._target_idx = 0       # loop back
                    # re-seed IK from current q
                    if self.mode == "ik":
                        self._precompute_ik(self.data.qpos[:7])
                self._start_next_move()

        elif self._phase == "move":
            if self._traj_idx < len(self._traj):
                q = self._traj[self._traj_idx, :7]
                self.data.ctrl[:7] = q
                self._traj_idx += 1
            else:
                self._phase    = "hold"
                self._hold_cnt = self.hold_steps
                self._target_idx += 1

    # ------------------------------------------------------------------
    #  Per-mode setup
    # ------------------------------------------------------------------

    def _precompute_ik(self, q_seed):
        """Solve IK for every Cartesian target; store joint solutions."""
        self._ik_sols = []
        q = q_seed.copy()
        for pos in self.targets:
            T = transform2mat(pos[0], pos[1], pos[2], np.pi, 0.0, 0.0)
            q_sol, info = self.kin.ik(T, q_init=q)
            self._ik_sols.append(q_sol[:7])
            if info["success"]:
                q = q_sol[:7]    # warm-start next solve
            print(f"[IK] target {pos} → success={info['success']}  "
                  f"err={info['error_norm']:.4f}")

    def _precompute_trajectory(self, q_home):
        """Build a single multi-waypoint trajectory for all waypoints."""
        waypoints = [q_home]
        for wp in self.targets:
            wp = np.array(wp)
            if wp.shape == (3,):          # Cartesian → IK
                T = transform2mat(wp[0], wp[1], wp[2], np.pi, 0.0, 0.0)
                q_sol, info = self.kin.ik(T, q_init=waypoints[-1])
                waypoints.append(q_sol[:7])
            else:                          # already joint config
                waypoints.append(wp[:7])

        self._wp_q     = waypoints       # for sphere drawing via FK
        self._wp_pos   = [self.kin.get_ee_position(q) for q in waypoints[1:]]
        result = self.tg.multi_waypoint(waypoints, segment_duration=1.5, dt=self.model.opt.timestep)
        self._full_traj   = result["positions"]  # (T_total, 7)
        self._full_t_idx  = 0

    # ------------------------------------------------------------------
    #  Move execution
    # ------------------------------------------------------------------

    def _start_next_move(self):
        """Build trajectory from current pose to the next target."""
        q_start = self.data.qpos[:7].copy()

        if self.mode == "fk":
            q_end = np.array(self.targets[self._target_idx])[:7]

        elif self.mode == "ik":
            q_end = self._ik_sols[self._target_idx]

        elif self.mode == "trajectory":
            # play the pre-computed trajectory continuously
            if not hasattr(self, "_full_traj") or self._full_traj is None:
                return
            # treat the whole trajectory as one "target"; clear trail on replay
            self._traj       = self._full_traj
            self._traj_idx   = 0
            self._ee_trail   = []
            self._phase      = "move"
            return

        else:
            return

        result = self.tg.cubic(
            q_start, q_end,
            duration = self.steps_per_move * self.model.opt.timestep,
            dt       = self.model.opt.timestep,
        )
        self._traj     = result["positions"]
        self._traj_idx = 0
        self._phase    = "move"

    # ------------------------------------------------------------------
    #  Marker drawing
    # ------------------------------------------------------------------

    def _draw_markers(self):
        positions, types, sizes, rgbas = [], [], [], []

        # EE position — bright green sphere
        ee_pos = self.get_body_position(self.ee_body)
        positions.append(ee_pos)
        types.append("sphere")
        sizes.append([0.025])
        rgbas.append([0.0, 1.0, 0.2, 0.9])

        if self.mode == "ik":
            # Current target — red sphere
            if self._target_idx < len(self.targets):
                positions.append(np.array(self.targets[self._target_idx]))
                types.append("sphere")
                sizes.append([0.03])
                rgbas.append([1.0, 0.15, 0.15, 0.85])

        elif self.mode == "trajectory":
            # All waypoints — coloured spheres
            if hasattr(self, "_wp_pos"):
                for k, wp in enumerate(self._wp_pos):
                    positions.append(wp)
                    types.append("sphere")
                    sizes.append([0.022])
                    col = _WAYPOINT_COLORS[k % len(_WAYPOINT_COLORS)]
                    rgbas.append(col)

        elif self.mode == "fk":
            # FK target from current config — blue marker
            if self._target_idx < len(self.targets):
                q_t = np.array(self.targets[self._target_idx])
                q_full = self.data.qpos.copy()
                q_full[:len(q_t)] = q_t
                t_pos = self.kin.get_ee_position(q_full)
                positions.append(t_pos)
                types.append("sphere")
                sizes.append([0.03])
                rgbas.append([0.2, 0.4, 1.0, 0.85])

        if positions:
            self.add_visual_geom(positions, types, sizes, rgbas)

        # EE trail — red line segments (trajectory mode only)
        if self.mode == "trajectory" and len(self._ee_trail) > 1:
            segs = list(zip(self._ee_trail[:-1], self._ee_trail[1:]))
            self.add_visual_lines(segs, width=3.0, rgba=[1.0, 0.15, 0.15, 0.9])
