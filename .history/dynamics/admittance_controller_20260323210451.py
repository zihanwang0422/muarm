import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import mujoco
from src.mujoco_viewer import MuJoCoViewer
from kinematic.panda_kinematics import PandaKinematics
from utils.transform import quat2rotmat

SCENE = "models/franka_emika_panda/scene_torque.xml"
PANDA = "models/franka_emika_panda/panda_motor.xml"

# Joint torque limits: joints 1-4 → ±87 Nm, joints 5-7 → ±12 Nm
TAU_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


class FrankaAdmittanceDemo(MuJoCoViewer):
    """Task-space admittance control for Franka Panda.

    Architecture (two loops):

    Outer loop — virtual dynamics (position only):
        M_d * ẍ_virt = F_ext - D_d * ẋ_virt - K_d * (x_virt - x_eq)

    where F_ext is the perturbation force read from xfrc_applied.
    Integrating this ODE gives a *virtual reference position* x_virt that
    compliantly follows the applied force.

    Inner loop — Cartesian PD + gravity compensation (torque control):
        F_ctrl = Kp_pos * (x_virt - x_ee)
               - Kd_pos * ẋ_ee[:3]
               + Kp_rot * e_rot
               - Kd_rot * ẋ_ee[3:]
        τ = J^T * F_ctrl + τ_bias

    Key difference from impedance control:
        Impedance:  measures displacement → outputs force  (τ = J^T K e)
        Admittance: measures force → outputs motion  (integrates virtual ODE,
                    inner PD tracks the moving virtual target)
    """

    # --- Virtual dynamics (mass-damper-spring on virtual EE position) ---
    # Tune M_d lower → more compliant / faster response
    # Tune D_d higher → smoother, more damped motion
    # Tune K_d  → spring that pulls virtual target back to equilibrium
    M_D = np.array([1.5, 1.5, 1.5])   # virtual mass (kg)
    D_D = np.array([25.0, 25.0, 25.0]) # virtual damping (N·s/m)
    K_D = np.array([30.0, 30.0, 30.0]) # virtual stiffness (N/m) — 0 = free drift

    # --- Inner Cartesian PD ---
    KP_POS = np.array([400.0, 400.0, 400.0])  # position stiffness (N/m)
    KD_POS = np.array([80.0,  80.0,  80.0])   # position damping (N·s/m)
    KP_ROT = np.array([40.0,  40.0,  40.0])   # orientation stiffness (Nm/rad)
    KD_ROT = np.array([8.0,   8.0,   8.0])    # orientation damping (Nm·s/rad)

    # Force dead-band — perturbations smaller than this are ignored (N)
    F_DEADBAND = 0.3

    def runBefore(self):
        home = self.model.key_qpos[0].copy()
        self.data.qpos[:] = home
        mujoco.mj_forward(self.model, self.data)

        self.kin   = PandaKinematics(ee_frame="hand")
        self.kin.build_from_mjcf(PANDA)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # Equilibrium pose
        self.eq_pos = self.data.body(self.ee_id).xpos.copy()
        self.eq_R   = quat2rotmat(self.data.body(self.ee_id).xquat.copy())

        # Virtual state: position and velocity of the virtual reference point
        self.x_virt  = self.eq_pos.copy()   # (3,)
        self.dx_virt = np.zeros(3)           # (3,)

        self._log_counter = 0

        print("\n" + "=" * 55)
        print(" Franka Task-space Admittance Control Demo")
        print(" 1. Double-click the EE body (hand)")
        print(" 2. Ctrl + drag to apply a force")
        print(" 3. The arm compliantly follows the force;")
        print("    spring K_D gently returns it to equilibrium.")
        print(" Tune M_D / D_D / K_D at the top of this file.")
        print("=" * 55 + "\n")

    def runFunc(self):
        dt = self.model.opt.timestep

        v7    = self.data.qvel[:7].copy()
        g_tau = self.data.qfrc_bias[:7].copy()   # gravity + Coriolis + centrifugal

        ee_pos  = self.data.body(self.ee_id).xpos.copy()
        ee_quat = self.data.body(self.ee_id).xquat.copy()
        R_cur   = quat2rotmat(ee_quat)

        # Jacobian (MuJoCo native, world frame)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J     = np.vstack([jacp, jacr])[:, :7]
        ee_vel = J @ v7   # 6D EE velocity [vx vy vz wx wy wz]

        # ------------------------------------------------------------------ #
        #  Outer loop: integrate virtual dynamics
        # ------------------------------------------------------------------ #
        # Read perturbation force from viewer drag (world frame, first 3 = force)
        F_ext_raw = self.data.xfrc_applied[self.ee_id][:3].copy()

        # Apply dead-band to avoid drift from numerical noise
        F_ext = np.where(np.abs(F_ext_raw) > self.F_DEADBAND, F_ext_raw, 0.0)

        # Virtual mass-damper-spring ODE (semi-implicit Euler)
        spring_force = self.K_D * (self.x_virt - self.eq_pos)
        ddx_virt = (F_ext - self.D_D * self.dx_virt - spring_force) / self.M_D

        self.dx_virt += ddx_virt * dt
        self.x_virt  += self.dx_virt * dt

        # ------------------------------------------------------------------ #
        #  Inner loop: Cartesian PD to track x_virt + hold orientation
        # ------------------------------------------------------------------ #
        # Position tracking
        e_pos = self.x_virt - ee_pos
        F_pos = self.KP_POS * e_pos - self.KD_POS * ee_vel[:3]

        # Orientation: hold equilibrium (no admittance on rotation)
        e_rot = 0.5 * (
            np.cross(R_cur[:, 0], self.eq_R[:, 0]) +
            np.cross(R_cur[:, 1], self.eq_R[:, 1]) +
            np.cross(R_cur[:, 2], self.eq_R[:, 2])
        )
        F_rot = self.KP_ROT * e_rot - self.KD_ROT * ee_vel[3:]

        F_cart   = np.concatenate([F_pos, F_rot])
        tau_ctrl = J.T @ F_cart

        # Final torque = inner PD + gravity compensation
        tau_final = tau_ctrl + g_tau
        self.data.ctrl[:7] = np.clip(tau_final, -TAU_LIMITS, TAU_LIMITS)

        # ------------------------------------------------------------------ #
        #  Periodic console log
        # ------------------------------------------------------------------ #
        self._log_counter += 1
        if self._log_counter % 250 == 0:
            disp = np.linalg.norm(self.x_virt - self.eq_pos)
            fmag = np.linalg.norm(F_ext)
            print(f" [Admittance] |F_ext|={fmag:.2f} N  "
                  f"virt_disp={disp*100:.1f} cm  "
                  f"|τ|={np.linalg.norm(tau_ctrl):.1f} Nm")

        # ------------------------------------------------------------------ #
        #  Visualise
        # ------------------------------------------------------------------ #
        self.handle.user_scn.ngeom = 0
        # Equilibrium = yellow, virtual target = cyan, current EE = green
        self.add_visual_geom(
            [self.eq_pos, self.x_virt, ee_pos],
            ["sphere", "sphere", "sphere"],
            [[0.03], [0.025], [0.02]],
            [[1.0, 0.8, 0.0, 0.7],   # yellow  — equilibrium
             [0.0, 0.9, 0.9, 0.7],   # cyan    — virtual target
             [0.0, 1.0, 0.0, 0.8]]   # green   — real EE
        )
        # Line: virtual target → real EE (tracking error)
        self.add_visual_lines(
            [(self.x_virt, ee_pos)],
            rgba=[0.2, 0.6, 1.0, 0.9]
        )
        # Line: equilibrium → virtual target (compliance displacement)
        if np.linalg.norm(self.x_virt - self.eq_pos) > 0.005:
            self.add_visual_lines(
                [(self.eq_pos, self.x_virt)],
                rgba=[1.0, 0.5, 0.0, 0.7]
            )


if __name__ == "__main__":
    FrankaAdmittanceDemo(SCENE, distance=2.5, azimuth=-45, elevation=-25).run_loop()
