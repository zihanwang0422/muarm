"""
SO-ARM100 Inverse Kinematics Module

Provides three IK solving methods:
1. Analytical IK (geometric decomposition)
2. Numerical IK (Jacobian pseudo-inverse / damped least squares)
3. Neural Network IK (learned from FK data)
"""

import numpy as np
from typing import List, Optional, Tuple
from .forward_kinematics import ForwardKinematics


class InverseKinematics:
    """
    Inverse Kinematics solver for the SO-ARM100 6-DOF robot arm.

    Supports analytical, numerical (Jacobian-based), and neural-network methods.
    Units: mm for position, radians for angles.
    """

    def __init__(self):
        self.fk = ForwardKinematics()
        self.num_joints = 6
        self._nn_model = None
        self._nn_scaler_X = None
        self._nn_scaler_y = None

    # ─────────────────────────────────────────────────────────────────
    # Analytical IK
    # ─────────────────────────────────────────────────────────────────

    def analytical(self, quaternion: np.ndarray, position: np.ndarray) -> List[float]:
        """
        Analytical IK using geometric decomposition.

        Strategy:
          1. Compute wrist center from target EE pose.
          2. Solve J1 (base rotation) from wrist center XY.
          3. Solve J2, J3 via planar 2-link IK.
          4. Solve J4, J5, J6 from wrist orientation.

        Falls back to numerical IK if analytical solution fails.

        Args:
            quaternion: Target quaternion [x, y, z, w].
            position: Target position [x, y, z] in mm.

        Returns:
            List of 6 joint angles in radians.
        """
        target_pos = np.asarray(position, dtype=float)
        R_target = self.fk.quaternion_to_rotation_matrix(quaternion)
        angles = np.zeros(6)

        try:
            # Wrist center: subtract EE + wrist offsets rotated by target orientation
            total_wrist_offset = np.array([
                0,
                self.fk.link_offsets[4][1] + self.fk.link_offsets[5][1] + self.fk.grab_offset[1],
                self.fk.link_offsets[4][2] + self.fk.link_offsets[5][2] + self.fk.grab_offset[2],
            ])
            wrist_center = target_pos - R_target @ total_wrist_offset

            # Joint 1 (Z-axis)
            angles[0] = np.arctan2(wrist_center[1], wrist_center[0])

            # Project into radial plane
            r_xy = np.sqrt(wrist_center[0] ** 2 + wrist_center[1] ** 2)
            p_y = r_xy - self.fk.link_offsets[1][1]
            p_z = wrist_center[2] - self.fk.link_offsets[1][2]

            # 2-link lengths
            L2 = np.sqrt(self.fk.link_offsets[2][1] ** 2 + self.fk.link_offsets[2][2] ** 2)
            L3 = np.sqrt(self.fk.link_offsets[3][1] ** 2 + self.fk.link_offsets[3][2] ** 2)
            alpha2 = np.arctan2(-self.fk.link_offsets[2][2], self.fk.link_offsets[2][1])
            alpha3 = np.arctan2(self.fk.link_offsets[3][2], self.fk.link_offsets[3][1])

            D = np.sqrt(p_y ** 2 + p_z ** 2)
            if D > L2 + L3 or D < abs(L2 - L3):
                raise ValueError("Target unreachable")

            cos_t3 = np.clip((D ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3), -1, 1)
            t3_geom = np.arccos(cos_t3)
            angles[2] = t3_geom - alpha2 - alpha3

            phi = np.arctan2(p_z, p_y)
            psi = np.arctan2(L3 * np.sin(t3_geom), L2 + L3 * np.cos(t3_geom))
            angles[1] = phi - psi + alpha2

            # Wrist orientation
            T01 = self.fk.transformation_matrix(angles[0], self.fk.rotation_axes[0], self.fk.link_offsets[0])
            T12 = self.fk.transformation_matrix(angles[1], self.fk.rotation_axes[1], self.fk.link_offsets[1])
            T23 = self.fk.transformation_matrix(angles[2], self.fk.rotation_axes[2], self.fk.link_offsets[2])
            T34 = self.fk.transformation_matrix(0, self.fk.rotation_axes[3], self.fk.link_offsets[3])
            R04 = (T01 @ T12 @ T23 @ T34)[:3, :3]
            R46 = R04.T @ R_target

            # Extract wrist Euler angles (X-Y-X for joints 4,5,6)
            angles[4] = np.arctan2(np.sqrt(R46[0, 1] ** 2 + R46[0, 2] ** 2), R46[0, 0])
            s5 = np.sin(angles[4])
            if abs(s5) > 1e-6:
                angles[3] = np.arctan2(R46[0, 1] / s5, R46[0, 2] / s5)
                angles[5] = np.arctan2(R46[1, 0] / s5, -R46[2, 0] / s5)
            else:
                angles[3] = 0.0
                angles[5] = np.arctan2(-R46[2, 1], R46[1, 1])

            # Quick FK check
            _, pos_check = self.fk.compute(angles)
            if np.linalg.norm(pos_check - target_pos) > 5.0:  # 5 mm tolerance
                raise ValueError("Analytical solution inaccurate, fallback to numerical")

        except Exception:
            return self.numerical(quaternion, position)

        return angles.tolist()

    # ─────────────────────────────────────────────────────────────────
    # Numerical IK (Jacobian-based)
    # ─────────────────────────────────────────────────────────────────

    def compute_jacobian(self, angles: List[float]) -> np.ndarray:
        """
        Compute the 6×6 Jacobian via numerical differentiation.

        Returns:
            6×6 Jacobian: rows 0-2 = position, rows 3-5 = orientation.
        """
        delta = 1e-6
        q_cur, p_cur = self.fk.compute(angles)
        R_cur = self.fk.quaternion_to_rotation_matrix(q_cur)

        J = np.zeros((6, 6))
        for i in range(6):
            a_pert = list(angles)
            a_pert[i] += delta
            q_pert, p_pert = self.fk.compute(a_pert)
            R_pert = self.fk.quaternion_to_rotation_matrix(q_pert)

            # Position Jacobian
            J[:3, i] = (p_pert - p_cur) / delta

            # Orientation Jacobian (axis-angle from rotation difference)
            R_diff = R_pert @ R_cur.T
            trace = np.trace(R_diff)
            ang = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            if ang > 1e-8:
                axis = np.array([
                    R_diff[2, 1] - R_diff[1, 2],
                    R_diff[0, 2] - R_diff[2, 0],
                    R_diff[1, 0] - R_diff[0, 1]
                ]) / (2 * np.sin(ang))
                J[3:, i] = axis * ang / delta
            else:
                J[3:, i] = 0.0

        return J

    @staticmethod
    def _rotation_error(R_desired: np.ndarray, R_current: np.ndarray) -> np.ndarray:
        """Compute orientation error as axis-angle vector."""
        R_err = R_desired @ R_current.T
        trace = np.trace(R_err)
        ang = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        if ang < 1e-8:
            return np.zeros(3)
        axis = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ]) / (2 * np.sin(ang))
        return axis * ang

    def numerical(
        self,
        quaternion: np.ndarray,
        position: np.ndarray,
        initial_angles: Optional[List[float]] = None,
        max_iterations: int = 200,
        tolerance: float = 1e-4,
        damping: float = 0.1,
    ) -> List[float]:
        """
        Numerical IK using damped least-squares (Levenberg-Marquardt).

        Args:
            quaternion: Target quaternion [x, y, z, w].
            position: Target position [x, y, z] in mm.
            initial_angles: Initial guess. Defaults to zeros.
            max_iterations: Max iterations.
            tolerance: Convergence tolerance (norm of error vector).
            damping: Regularization factor λ.

        Returns:
            List of 6 joint angles in radians.
        """
        target_pos = np.asarray(position, dtype=float)
        R_target = self.fk.quaternion_to_rotation_matrix(quaternion)

        angles = np.array(initial_angles if initial_angles else [0.0] * 6, dtype=float)

        for _ in range(max_iterations):
            q_cur, p_cur = self.fk.compute(angles)
            R_cur = self.fk.quaternion_to_rotation_matrix(q_cur)

            pos_err = target_pos - p_cur
            ori_err = self._rotation_error(R_target, R_cur)
            error = np.concatenate([pos_err, ori_err])

            if np.linalg.norm(error) < tolerance:
                break

            J = self.compute_jacobian(angles)
            JJT = J @ J.T + damping * np.eye(6)
            try:
                J_pinv = J.T @ np.linalg.inv(JJT)
            except np.linalg.LinAlgError:
                J_pinv = np.linalg.pinv(J)

            angles += J_pinv @ error

        # Normalize to [-π, π]
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        return angles.tolist()

    # ─────────────────────────────────────────────────────────────────
    # Neural Network IK
    # ─────────────────────────────────────────────────────────────────

    def load_nn_model(self, model_path: str, scaler_path: str):
        """
        Load a pre-trained Keras NN model and its scalers.

        Args:
            model_path: Path to .keras model file.
            scaler_path: Path to .pkl scaler file (mean_X, std_X, mean_y, std_y).
        """
        import pickle
        from tensorflow import keras

        self._nn_model = keras.models.load_model(model_path)

        with open(scaler_path, "rb") as f:
            mean_X, std_X, mean_y, std_y = pickle.load(f)

        self._nn_scaler_X = (mean_X, std_X)
        self._nn_scaler_y = (mean_y, std_y)

    def neural_network(
        self,
        quaternion: np.ndarray,
        position: np.ndarray,
        reference_angles: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Predict joint angles using a pre-trained neural network.

        Input features: [6 reference joints, 4 quaternion, 3 position] = 13.
        Output: 6 target joint angles.

        Args:
            quaternion: Target quaternion [x, y, z, w].
            position: Target position [x, y, z] in mm.
            reference_angles: Current/reference joint angles (6). Defaults to zeros.

        Returns:
            List of 6 predicted joint angles in radians.
        """
        if self._nn_model is None:
            raise RuntimeError("NN model not loaded. Call load_nn_model() first.")

        if reference_angles is None:
            reference_angles = [0.0] * 6

        # Build feature vector: [ref_joints(6), quat(4), pos(3)]
        features = np.array(
            list(reference_angles) + list(quaternion) + list(position),
            dtype=float,
        ).reshape(1, -1)

        # Normalize
        mean_X, std_X = self._nn_scaler_X
        features_scaled = (features - mean_X) / std_X

        # Predict
        pred_scaled = self._nn_model.predict(features_scaled, verbose=0)

        # De-normalize
        mean_y, std_y = self._nn_scaler_y
        pred = pred_scaled * std_y + mean_y

        return pred.flatten().tolist()


if __name__ == "__main__":
    ik = InverseKinematics()
    fk = ForwardKinematics()

    # Test: FK → IK → FK round-trip
    test_angles = [0.3, 1.2, -1.5, -0.8, 0.2, 0.1]
    quat, pos = fk.compute(test_angles)
    print(f"Original angles (deg): {np.rad2deg(test_angles)}")
    print(f"FK position (mm): {pos}")
    print(f"FK quaternion: {quat}")

    # Numerical IK
    recovered = ik.numerical(quat, pos)
    q2, p2 = fk.compute(recovered)
    print(f"\nNumerical IK angles (deg): {np.rad2deg(recovered)}")
    print(f"Recovered position (mm): {p2}")
    print(f"Position error (mm): {np.linalg.norm(p2 - pos):.4f}")

    # Analytical IK
    recovered_a = ik.analytical(quat, pos)
    q3, p3 = fk.compute(recovered_a)
    print(f"\nAnalytical IK angles (deg): {np.rad2deg(recovered_a)}")
    print(f"Recovered position (mm): {p3}")
    print(f"Position error (mm): {np.linalg.norm(p3 - pos):.4f}")
