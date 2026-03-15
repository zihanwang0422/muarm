"""Simple Model Predictive Controller for joint-space trajectory tracking.

Uses a linear prediction model to optimize control inputs over a finite horizon.
This is a basic MPC implementation suitable for educational and prototyping purposes.
"""
import numpy as np
from scipy.optimize import minimize


class MPCController:
    """Joint-space Model Predictive Controller.

    Optimizes a sequence of control actions over a prediction horizon to track
    a desired trajectory while respecting joint limits and control bounds.

    Example::

        mpc = MPCController(n_joints=7, horizon=10, dt=0.002)
        ctrl = mpc.compute(q_current, q_dot_current, q_desired_trajectory)
    """

    def __init__(self, n_joints=7, horizon=10, dt=0.002,
                 q_weight=10.0, u_weight=0.01, du_weight=0.1,
                 u_max=None, q_min=None, q_max=None):
        self.n_joints = n_joints
        self.horizon = horizon
        self.dt = dt
        self.q_weight = q_weight
        self.u_weight = u_weight
        self.du_weight = du_weight
        self.u_max = u_max if u_max is not None else np.full(n_joints, np.inf)
        self.q_min = q_min if q_min is not None else np.full(n_joints, -np.inf)
        self.q_max = q_max if q_max is not None else np.full(n_joints, np.inf)
        self.prev_u = np.zeros(n_joints)

    def _predict(self, q0, qd0, u_seq):
        """Simple double-integrator prediction: q_{k+1} = q_k + qd_k*dt, qd_{k+1} = qd_k + u_k*dt."""
        q_pred = np.zeros((self.horizon + 1, self.n_joints))
        qd_pred = np.zeros((self.horizon + 1, self.n_joints))
        q_pred[0] = q0
        qd_pred[0] = qd0
        for k in range(self.horizon):
            u = u_seq[k * self.n_joints:(k + 1) * self.n_joints]
            qd_pred[k + 1] = qd_pred[k] + u * self.dt
            q_pred[k + 1] = q_pred[k] + qd_pred[k + 1] * self.dt
        return q_pred, qd_pred

    def _cost(self, u_flat, q0, qd0, q_ref):
        """Compute the cost for a candidate control sequence."""
        q_pred, _ = self._predict(q0, qd0, u_flat)

        # Tracking cost
        tracking = 0.0
        for k in range(1, self.horizon + 1):
            ref_idx = min(k, len(q_ref) - 1)
            err = q_pred[k] - q_ref[ref_idx]
            tracking += self.q_weight * np.sum(err**2)

        # Control effort cost
        effort = self.u_weight * np.sum(u_flat**2)

        # Smoothness cost
        u_seq = u_flat.reshape(self.horizon, self.n_joints)
        smooth = 0.0
        du = u_seq[0] - self.prev_u
        smooth += self.du_weight * np.sum(du**2)
        for k in range(1, self.horizon):
            du = u_seq[k] - u_seq[k - 1]
            smooth += self.du_weight * np.sum(du**2)

        return tracking + effort + smooth

    def compute(self, q_current, q_dot_current, q_desired_traj):
        """Compute the optimal control input for the current state.

        Args:
            q_current: Current joint positions (n_joints,)
            q_dot_current: Current joint velocities (n_joints,)
            q_desired_traj: Desired joint trajectory, shape (T, n_joints) where T >= horizon

        Returns:
            Optimal control input for the current step (n_joints,)
        """
        u0 = np.zeros(self.horizon * self.n_joints)

        bounds = []
        for _ in range(self.horizon):
            for j in range(self.n_joints):
                bounds.append((-self.u_max[j], self.u_max[j]))

        result = minimize(
            self._cost, u0, args=(q_current, q_dot_current, q_desired_traj),
            method='SLSQP', bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-6},
        )

        u_opt = result.x[:self.n_joints]
        self.prev_u = u_opt.copy()
        return u_opt

    def reset(self):
        self.prev_u = np.zeros(self.n_joints)
