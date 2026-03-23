"""Panda robot kinematics: FK, IK, and Jacobian computation.

Uses Pinocchio for FK, Jacobian, and inverse dynamics.
IK is solved numerically via iterative damped-Jacobian (no CasADi required).
"""
import numpy as np

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False


def _damped_pinv(jacobian, damping=0.05):
    """Compute damped pseudo-inverse for numerical stability."""
    j = np.asarray(jacobian, dtype=np.float64)
    m, n = j.shape
    if m >= n:
        return np.linalg.solve(j.T @ j + (damping ** 2) * np.eye(n), j.T)
    return j.T @ np.linalg.solve(j @ j.T + (damping ** 2) * np.eye(m), np.eye(m))


def _rot_to_rpy(rotation):
    """Convert rotation matrix to roll-pitch-yaw."""
    sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


class PandaKinematics:
    """FK/IK solver for a Franka Emika Panda robot (7-DOF).

    Supports loading from MJCF or URDF via Pinocchio. Provides FK, IK,
    and Jacobian computation.

    IK uses iterative damped Jacobian (no CasADi required).

    Example::

        kin = PandaKinematics()
        kin.build_from_mjcf("models/franka_emika_panda/panda.xml")
        T = transform2mat(0.5, 0.0, 0.3, np.pi, 0, 0)
        q, info = kin.ik(T)
        print("FK verification:", kin.fk(q))
    """

    def __init__(self, ee_frame="link7"):
        """Initialize kinematics solver.

        Args:
            ee_frame: Name of the end-effector frame in the model.
                      Use "link7" for the standard Panda flange.
        """
        self.frame_name = ee_frame
        self.model = None
        self.data = None
        self.ee_id = None
        self._init_q = None
        self._solver_ready = False

    def build_from_mjcf(self, mjcf_file):
        """Build the kinematic model from a MuJoCo MJCF file."""
        if not HAS_PINOCCHIO:
            raise RuntimeError("Pinocchio is required for kinematics. Install with: pip install pin")
        robot = pin.RobotWrapper.BuildFromMJCF(mjcf_file)
        self._setup(robot)

    def build_from_urdf(self, urdf_file):
        """Build the kinematic model from a URDF file."""
        if not HAS_PINOCCHIO:
            raise RuntimeError("Pinocchio is required for kinematics. Install with: pip install pin")
        robot = pin.RobotWrapper.BuildFromURDF(urdf_file)
        self._setup(robot)

    def _setup(self, robot):
        self.model = robot.model
        self.data = robot.data
        self.ee_id = self.model.getFrameId(self.frame_name)
        # Use joint-range midpoint as default init — pin.neutral() returns zeros
        # which places joint4 (range [-3.07, -0.07]) outside its valid range.
        lo = self.model.lowerPositionLimit
        hi = self.model.upperPositionLimit
        self._init_q = 0.5 * (lo + hi)
        # Number of active arm joints (exclude finger slides at the end)
        self.n_arm = 7
        self._solver_ready = True

    def fk(self, q):
        """Compute forward kinematics.

        Args:
            q: Joint angles (model.nq,)

        Returns:
            4x4 homogeneous transformation matrix of the end-effector.
        """
        q = np.array(q, dtype=np.float64)
        if len(q) < self.model.nq:
            q_full = self._init_q.copy()
            q_full[:len(q)] = q
            q = q_full
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        se3 = self.data.oMf[self.ee_id]
        T = np.eye(4)
        T[:3, :3] = se3.rotation
        T[:3, 3] = se3.translation
        return T

    def ik(self, target_tf, q_init=None,
           max_iters=200, tol=1e-4, alpha=0.5, lambda_d=0.05):
        """Compute inverse kinematics using damped Jacobian iterations.

        Args:
            target_tf: 4x4 target homogeneous transformation matrix
            q_init: Initial joint angle guess. If None, uses last solution.
            max_iters: Maximum number of iterations
            tol: Convergence tolerance on the 6D error norm
            alpha: Step size (0 < alpha <= 1)
            lambda_d: Damping coefficient for the pseudoinverse

        Returns:
            (q_solution, info_dict) where info_dict contains:
                - "success": bool
                - "iterations": int
                - "error_norm": float
                - "sol_tauff": feedforward torques from inverse dynamics
        """
        if not self._solver_ready:
            raise RuntimeError("IK solver not initialized. Call build_from_mjcf/urdf first.")

        q = np.array(q_init, dtype=np.float64) if q_init is not None else self._init_q.copy()
        # Pad to full model.nq if only arm joints were provided
        if len(q) < self.model.nq:
            q_full = self._init_q.copy()
            q_full[:len(q)] = q
            q = q_full
        q_lo = self.model.lowerPositionLimit
        q_hi = self.model.upperPositionLimit
        n = self.n_arm  # only update the 7 arm joints

        R_des = target_tf[:3, :3]
        p_des = target_tf[:3, 3]

        err_norm = np.inf
        for i in range(max_iters):
            # FK
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            se3 = self.data.oMf[self.ee_id]

            # Position error
            e_pos = p_des - se3.translation

            # Orientation error via rotation matrix log
            R_err = R_des @ se3.rotation.T
            e_rot = pin.log3(R_err)

            e = np.concatenate([e_pos, e_rot])
            err_norm = np.linalg.norm(e)
            if err_norm < tol:
                break

            # Jacobian — only use arm-joint columns
            J_full = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_id, pin.ReferenceFrame.WORLD
            )
            J = J_full[:, :n]
            J_pinv = _damped_pinv(J, lambda_d)
            dq = alpha * J_pinv @ e

            q[:n] = np.clip(q[:n] + dq, q_lo[:n], q_hi[:n])

        self._init_q = q.copy()
        success = err_norm < tol * 10

        v = np.zeros(self.model.nv)
        tauff = pin.rnea(self.model, self.data, q, v, np.zeros(self.model.nv))

        return q, {
            "success": success,
            "iterations": i + 1,
            "error_norm": err_norm,
            "sol_tauff": tauff,
        }

    def jacobian(self, q):
        """Compute the 6xN frame Jacobian at the EE in world frame.

        Args:
            q: Joint angles (model.nq,)

        Returns:
            6xN Jacobian matrix (linear velocity rows 0-2, angular rows 3-5).
        """
        q = np.array(q, dtype=np.float64)
        if len(q) < self.model.nq:
            q_full = self._init_q.copy()
            q_full[:len(q)] = q
            q = q_full
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J_full = pin.computeFrameJacobian(
            self.model, self.data, q, self.ee_id, pin.ReferenceFrame.WORLD
        )
        return J_full[:, :self.n_arm]   # (6, 7) — arm joints only

    def get_ee_position(self, q):
        """Return the end-effector [x, y, z] position for given joint angles."""
        return self.fk(q)[:3, 3]

    def get_ee_pose(self, q):
        """Return (x, y, z, roll, pitch, yaw) for given joint angles."""
        tf = self.fk(q)
        xyz = tf[:3, 3]
        rpy = _rot_to_rpy(tf[:3, :3])
        return np.concatenate([xyz, rpy])
