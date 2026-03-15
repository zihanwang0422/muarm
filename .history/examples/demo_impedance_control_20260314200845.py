"""Impedance drag-and-return demo.

Float mode  — gravity compensation only.  The robot is freely backdrivable.
              Use MuJoCo's built-in mouse drag (double-click the hand body,
              then Ctrl+drag) to displace the end-effector.

Return mode — Cartesian impedance drives the EE back to the equilibrium pose
              recorded at startup.  Triggers automatically the moment you
              release the mouse after dragging.

Visual:
  Yellow sphere  — equilibrium (fixed)
  Green  sphere  — live EE
  Red    line    — displacement vector (Return phase only)

HOW TO DRAG in the MuJoCo viewer
  1. Double-click on the robot's hand body to select it.
  2. Hold Ctrl and drag with the left mouse button.
  3. Release → the robot returns to start.

Run:
    python examples/demo_impedance_control.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import mujoco
import pinocchio as pin

from src.mujoco_viewer import MuJoCoViewer
from kinematic.panda_kinematics import PandaKinematics
from utils.transform import quat2rotmat

SCENE = "models/franka_emika_panda/scene.xml"
PANDA = "models/franka_emika_panda/panda.xml"


class ImpedanceDragDemo(MuJoCoViewer):
    # Cartesian impedance gains (Return phase)
    KP = np.array([300, 300, 300,  25,  25,  25], dtype=float)
    KD = np.array([ 30,  30,  30,   2,   2,   2], dtype=float)

    # Position actuator gains from panda.xml gainprm
    ACT_KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000], dtype=float)

    # Thresholds
    DRAG_DETECT_F  = 1.0   # N  — xfrc_applied magnitude to count as active drag
    RETURN_TRIGGER = 0.04  # m  — minimum displacement to trigger Return phase
    RETURN_DONE    = 0.015 # m  — displacement considered "returned"

    def runBefore(self):
        home = self.model.key_qpos[0].copy()
        self.data.qpos[:] = home
        self.data.ctrl[:] = self.model.key_ctrl[0].copy()
        mujoco.mj_forward(self.model, self.data)

        # kinematics (Jacobian)
        self.kin = PandaKinematics(ee_frame="hand")
        self.kin.build_from_mjcf(PANDA)
        self.ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # Pinocchio for gravity compensation
        robot = pin.RobotWrapper.BuildFromMJCF(PANDA)
        self.pin_model = robot.model
        self.pin_data  = robot.data

        # Record equilibrium EE pose
        self.eq_pos = self.data.body(self.ee_id).xpos.copy()
        self.eq_R   = quat2rotmat(self.data.body(self.ee_id).xquat.copy())

        self._phase       = "float"
        self._was_dragging = False

        print("\n" + "="*55)
        print(" Impedance drag-and-return demo")
        print("="*55)
        print(" 1. Double-click on the hand (EE) body to select it")
        print(" 2. Hold Ctrl + left-drag to displace the arm")
        print(" 3. Release → robot returns to start position")
        print("="*55 + "\n")

    def _grav_comp(self, q7, v7):
        q9 = np.concatenate([q7, np.zeros(2)])
        v9 = np.concatenate([v7, np.zeros(2)])
        tau9 = pin.rnea(self.pin_model, self.pin_data, q9, v9, np.zeros(9))
        return tau9[:7]

    def runFunc(self):
        q7 = self.data.qpos[:7].copy()
        v7 = self.data.qvel[:7].copy()

        g_tau = self._grav_comp(q7, v7)

        ee_pos  = self.data.body(self.ee_id).xpos.copy()
        ee_quat = self.data.body(self.ee_id).xquat.copy()
        R_cur   = quat2rotmat(ee_quat)
        J       = self.kin.jacobian(q7)
        ee_vel  = J @ v7
        dist    = np.linalg.norm(ee_pos - self.eq_pos)

        # Detect active drag from MuJoCo's built-in Ctrl+drag perturbation
        f_ext       = np.linalg.norm(self.data.xfrc_applied[self.ee_id])
        is_dragging = f_ext > self.DRAG_DETECT_F

        # ── FLOAT phase: gravity comp, robot freely backdrivable ────────
        if self._phase == "float":
            self.data.ctrl[:7] = g_tau

            # Trigger Return when user releases after displacing the EE
            if self._was_dragging and not is_dragging and dist > self.RETURN_TRIGGER:
                self._phase = "return"
                print(f"  [Return]  EE offset {dist*100:.1f} cm — returning…")

        # ── RETURN phase: Cartesian impedance back to equilibrium ───────
        else:
            e_pos  = self.eq_pos - ee_pos
            e_rot  = pin.log3(self.eq_R @ R_cur.T)
            F_imp  = self.KP * np.concatenate([e_pos, e_rot]) - self.KD * ee_vel
            tau_imp = np.clip(J.T @ F_imp, -80, 80)
            ctrl    = q7 + (g_tau + tau_imp) / self.ACT_KP

            if dist < self.RETURN_DONE:
                self._phase = "float"
                print("  [Float]   Returned — ready to drag again")

        self._was_dragging = is_dragging

        # ── apply control ───────────────────────────────────────────────
        self.data.ctrl[:7] = np.clip(
            ctrl,
            self.model.jnt_range[:7, 0],
            self.model.jnt_range[:7, 1],
        )

        # ── visual overlay ──────────────────────────────────────────────
        self.handle.user_scn.ngeom = 0
        eq_alpha = 0.95 if self._phase == "return" else 0.55
        self.add_visual_geom(
            [self.eq_pos,                      ee_pos],
            ["sphere",                         "sphere"],
            [[0.035],                           [0.025]],
            [[1.0, 0.75, 0.0, eq_alpha],        [0.0, 1.0, 0.2, 0.9]],
        )
        if self._phase == "return":
            self.add_visual_lines(
                [(ee_pos, self.eq_pos)],
                width=4.0,
                rgba=[1.0, 0.2, 0.2, 0.9],
            )


ImpedanceDragDemo(SCENE, distance=2.5, azimuth=-45, elevation=-25).run_loop()
