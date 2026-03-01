"""
SO-ARM100 Inverse Kinematics Module

Provides two IK methods for the 5-DOF arm (excluding Jaw):
  1. Numerical IK — Jacobian damped least-squares (Levenberg-Marquardt)
  2. Neural Network IK — Learned mapping from (reference_joints, target_pose) → joints

Uses the same FK model as forward_kinematics.py.
"""

import numpy as np
from typing import List, Optional, Tuple
from .forward_kinematics import ForwardKinematics


class InverseKinematics:
    """
    Inverse Kinematics solver for SO-ARM100 (5-DOF arm joints).
    """

    def __init__(self):
        self.fk = ForwardKinematics()
        self.num_joints = ForwardKinematics.NUM_JOINTS
        self._nn_model = None
        self._nn_scaler = None

    # ─── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _rotation_error(R_desired: np.ndarray, R_current: np.ndarray) -> np.ndarray:
        """Orientation error as axis-angle vector."""
        R_err = R_desired @ R_current.T
        trace = np.clip(np.trace(R_err), -1.0, 3.0)
        ang = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        if ang < 1e-8:
            return np.zeros(3)
        axis = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ]) / (2 * np.sin(ang) + 1e-12)
        return axis * ang

    def _compute_jacobian(self, angles: np.ndarray) -> np.ndarray:
        """
        Numerical Jacobian (6×N) via finite differences.
        Rows 0-2: position, rows 3-5: orientation.
        """
        delta = 1e-6
        p0, q0 = self.fk.compute(angles)
        R0 = self.fk.quat_to_rotmat(q0)

        J = np.zeros((6, self.num_joints))
        for i in range(self.num_joints):
            a = angles.copy()
            a[i] += delta
            p1, q1 = self.fk.compute(a)
            R1 = self.fk.quat_to_rotmat(q1)

            J[:3, i] = (p1 - p0) / delta

            R_diff = R1 @ R0.T
            tr = np.clip(np.trace(R_diff), -1.0, 3.0)
            ang = np.arccos(np.clip((tr - 1) / 2, -1, 1))
            if ang > 1e-8:
                axis = np.array([R_diff[2,1]-R_diff[1,2],
                                 R_diff[0,2]-R_diff[2,0],
                                 R_diff[1,0]-R_diff[0,1]]) / (2*np.sin(ang))
                J[3:, i] = axis * ang / delta
        return J

    # ─── Numerical IK ───────────────────────────────────────────────

    def numerical(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        initial_angles: Optional[List[float]] = None,
        max_iter: int = 300,
        tol: float = 1e-4,
        damping: float = 0.05,
        pos_weight: float = 1.0,
        ori_weight: float = 0.3,
    ) -> List[float]:
        """
        Numerical IK using damped least-squares.

        Args:
            target_pos:       Target [x,y,z] in metres.
            target_quat_wxyz: Target quaternion [w,x,y,z].
            initial_angles:   Starting guess (5 values). Defaults to zeros.
            max_iter:         Maximum iterations.
            tol:              Convergence tolerance on weighted error norm.
            damping:          LM damping λ.
            pos_weight:       Weight for position error.
            ori_weight:       Weight for orientation error.

        Returns:
            List of 5 joint angles in radians.
        """
        R_target = self.fk.quat_to_rotmat(target_quat_wxyz)
        target_pos = np.asarray(target_pos, dtype=float)

        angles = np.array(initial_angles if initial_angles else [0.0]*self.num_joints, dtype=float)

        W = np.diag([pos_weight]*3 + [ori_weight]*3)

        for _ in range(max_iter):
            p_cur, q_cur = self.fk.compute(angles)
            R_cur = self.fk.quat_to_rotmat(q_cur)

            e_pos = target_pos - p_cur
            e_ori = self._rotation_error(R_target, R_cur)
            e = np.concatenate([e_pos, e_ori])

            if np.linalg.norm(W @ e) < tol:
                break

            J = self._compute_jacobian(angles)
            Jw = W @ J
            ew = W @ e
            JJT = Jw @ Jw.T + damping * np.eye(6)
            try:
                d_angles = Jw.T @ np.linalg.solve(JJT, ew)
            except np.linalg.LinAlgError:
                d_angles = np.linalg.pinv(J) @ e

            angles += d_angles

            # Clamp to joint limits
            for j in range(self.num_joints):
                lo, hi = self.fk.JOINT_LIMITS[j]
                angles[j] = np.clip(angles[j], lo, hi)

        return angles.tolist()

    # ─── Neural Network IK ──────────────────────────────────────────

    def load_nn_model(self, model_path: str, scaler_path: str):
        """
        Load pre-trained Keras model + scalers.

        Args:
            model_path:  Path to .keras file.
            scaler_path: Path to .pkl with (mean_X, std_X, mean_y, std_y).
        """
        import pickle
        from tensorflow import keras

        self._nn_model = keras.models.load_model(model_path)
        with open(scaler_path, "rb") as f:
            self._nn_scaler = pickle.load(f)  # (mean_X, std_X, mean_y, std_y)

    def neural_network(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        reference_angles: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Predict joint angles via a pre-trained NN.

        Features: [ref_joints(5), quat(4), pos(3)] = 12.
        Output: 5 target joint angles.
        """
        if self._nn_model is None:
            raise RuntimeError("NN model not loaded. Call load_nn_model() first.")
        if reference_angles is None:
            reference_angles = [0.0] * self.num_joints

        feat = np.array(list(reference_angles) + list(target_quat_wxyz) + list(target_pos),
                        dtype=float).reshape(1, -1)
        mean_X, std_X, mean_y, std_y = self._nn_scaler
        pred = self._nn_model.predict((feat - mean_X) / std_X, verbose=0)
        return (pred * std_y + mean_y).flatten().tolist()


if __name__ == "__main__":
    fk = ForwardKinematics()
    ik = InverseKinematics()

    test = [0.3, -1.0, 1.2, 0.5, -0.2]
    p, q = fk.compute(test)
    print(f"GT angles (deg): {np.rad2deg(test)}")
    print(f"FK pos: {p}  quat: {q}")

    recovered = ik.numerical(p, q)
    p2, q2 = fk.compute(recovered)
    print(f"IK angles (deg): {np.rad2deg(recovered)}")
    print(f"Pos error: {np.linalg.norm(p2-p)*1000:.4f} mm")
