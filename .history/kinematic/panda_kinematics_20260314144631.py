"""Panda robot kinematics: FK, IK, and Jacobian computation.

Uses Pinocchio + CasADi for optimization-based IK and analytical FK.
Falls back to MuJoCo-native FK when Pinocchio is not available.
"""
import numpy as np

try:
    import pinocchio as pin
    from pinocchio import casadi as cpin
    import casadi
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.transform import transform2mat, mat2transform


class PandaKinematics:
    """FK/IK solver for a Franka Emika Panda robot (7-DOF).

    Supports loading from MJCF or URDF via Pinocchio. Provides FK, IK,
    and Jacobian computation.

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
        self._solver_ready = False

    def build_from_mjcf(self, mjcf_file):
        """Build the kinematic model from a MuJoCo MJCF file."""
        if not HAS_PINOCCHIO:
            raise RuntimeError("Pinocchio is required for kinematics. Install with: pip install pin")
        robot = pin.RobotWrapper.BuildFromMJCF(mjcf_file)
        self.model = robot.model
        self.data = robot.data
        self._create_ik_solver()

    def build_from_urdf(self, urdf_file):
        """Build the kinematic model from a URDF file."""
        if not HAS_PINOCCHIO:
            raise RuntimeError("Pinocchio is required for kinematics. Install with: pip install pin")
        robot = pin.RobotWrapper.BuildFromURDF(urdf_file)
        self.model = robot.model
        self.data = robot.data
        self._create_ik_solver()

    def _create_ik_solver(self):
        """Set up the CasADi-based IK optimization problem."""
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.ee_id = self.model.getFrameId(self.frame_name)

        self._trans_err = casadi.Function(
            "trans_err", [self.cq, self.cTf],
            [self.cdata.oMf[self.ee_id].translation - self.cTf[:3, 3]],
        )
        self._rot_err = casadi.Function(
            "rot_err", [self.cq, self.cTf],
            [cpin.log3(self.cdata.oMf[self.ee_id].rotation @ self.cTf[:3, :3].T)],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.var_q_last = self.opti.parameter(self.model.nq)
        self.param_tf = self.opti.parameter(4, 4)

        trans_cost = casadi.sumsqr(self._trans_err(self.var_q, self.param_tf))
        rot_cost = casadi.sumsqr(self._rot_err(self.var_q, self.param_tf))
        reg_cost = casadi.sumsqr(self.var_q)
        smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit,
            self.var_q,
            self.model.upperPositionLimit,
        ))
        self.opti.minimize(
            20.0 * trans_cost + 0.01 * rot_cost + 0.005 * smooth_cost
        )

        opts = {
            'ipopt': {'print_level': 0, 'max_iter': 1000, 'tol': 1e-6},
            'print_time': False,
            'calc_lam_p': False,
        }
        self.opti.solver("ipopt", opts)
        self._init_q = np.zeros(self.model.nq)
        self._solver_ready = True

    def fk(self, q):
        """Compute forward kinematics.

        Args:
            q: Joint angles (model.nq,)

        Returns:
            4x4 homogeneous transformation matrix of the end-effector.
        """
        q = np.array(q, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        se3 = self.data.oMf[self.ee_id]
        T = np.eye(4)
        T[:3, :3] = se3.rotation
        T[:3, 3] = se3.translation
        return T

    def ik(self, target_tf, q_init=None):
        """Compute inverse kinematics using CasADi/IPOPT optimization.

        Args:
            target_tf: 4x4 target homogeneous transformation matrix
            q_init: Initial joint angle guess. If None, uses last solution.

        Returns:
            (q_solution, info_dict) where info_dict contains:
                - "success": bool
                - "sol_tauff": feedforward torques from inverse dynamics
        """
        if not self._solver_ready:
            raise RuntimeError("IK solver not initialized. Call build_from_mjcf/urdf first.")

        if q_init is not None:
            self._init_q = np.array(q_init)

        self.opti.set_initial(self.var_q, self._init_q)
        self.opti.set_value(self.param_tf, target_tf)
        self.opti.set_value(self.var_q_last, self._init_q)

        try:
            sol = self.opti.solve()
            sol_q = self.opti.value(self.var_q)
            self._init_q = sol_q.copy()

            v = np.zeros(self.model.nv)
            tauff = pin.rnea(self.model, self.data, sol_q, v, np.zeros(self.model.nv))

            dof = np.zeros(self.model.nq)
            dof[:len(sol_q)] = sol_q
            return dof, {"success": True, "sol_tauff": tauff}

        except Exception as e:
            sol_q = self.opti.debug.value(self.var_q)
            self._init_q = sol_q.copy()
            dof = np.zeros(self.model.nq)
            dof[:len(sol_q)] = sol_q
            return dof, {"success": False, "error": str(e), "sol_tauff": np.zeros(self.model.nv)}

    def jacobian(self, q):
        """Compute the 6xN frame Jacobian at the EE in world frame.

        Args:
            q: Joint angles (model.nq,)

        Returns:
            6xN Jacobian matrix (linear velocity rows 0-2, angular rows 3-5).
        """
        q = np.array(q, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return pin.computeFrameJacobian(
            self.model, self.data, q, self.ee_id, pin.ReferenceFrame.WORLD
        )

    def get_ee_position(self, q):
        """Return the end-effector [x, y, z] position for given joint angles."""
        T = self.fk(q)
        return T[:3, 3]

    def get_ee_pose(self, q):
        """Return (x, y, z, roll, pitch, yaw) for given joint angles."""
        T = self.fk(q)
        return mat2transform(T)
