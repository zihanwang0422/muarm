"""PID Controller for joint-level or task-space control."""
import numpy as np


class PIDController:
    """Multi-axis PID controller.

    Can be used for single joints or as a vector PID for all joints simultaneously.

    Example::

        pid = PIDController(kp=100.0, ki=0.0, kd=10.0, n_dims=7)
        torque = pid.compute(error=q_desired - q_current, dt=0.001)
    """

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, n_dims=1):
        self.n_dims = n_dims
        self.kp = np.full(n_dims, kp) if np.isscalar(kp) else np.array(kp)
        self.ki = np.full(n_dims, ki) if np.isscalar(ki) else np.array(ki)
        self.kd = np.full(n_dims, kd) if np.isscalar(kd) else np.array(kd)
        self.integral = np.zeros(n_dims)
        self.prev_error = np.zeros(n_dims)

    def compute(self, error, dt):
        """Compute PID output given the current error and timestep."""
        error = np.atleast_1d(error)
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else np.zeros(self.n_dims)
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error.copy()
        return output

    def reset(self):
        """Reset integrator and previous error."""
        self.integral = np.zeros(self.n_dims)
        self.prev_error = np.zeros(self.n_dims)
