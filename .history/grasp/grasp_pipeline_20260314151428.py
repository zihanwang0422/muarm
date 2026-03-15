"""Basic vision-based grasping pipeline for Panda robot.

A simple detect-approach-grasp pipeline using simulated camera input
and basic centroid detection. Designed as a starting point that can be
extended with more sophisticated perception (e.g., deep learning models).
"""
import numpy as np
import mujoco

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.mujoco_viewer import MuJoCoViewer
from grasp.camera import SimCamera
from utils.transform import transform2mat

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class GraspPipeline(MuJoCoViewer):
    """Basic camera-guided grasping pipeline for Panda robot.

    Steps:
    1. Capture image from overhead camera
    2. Detect object position (centroid-based or color-based)
    3. Plan approach trajectory (pre-grasp -> grasp -> lift)
    4. Execute open-loop grasp

    This is a basic pipeline for educational purposes. For real applications,
    replace the detection step with a proper perception module.

    Example::

        pipeline = GraspPipeline(
            scene_xml="models/franka_emika_panda/scene.xml",
            camera_name="overhead_cam"
        )
        pipeline.run_loop()
    """

    def __init__(self, scene_xml, camera_name=None,
                 distance=3, azimuth=-45, elevation=-30):
        super().__init__(scene_xml, distance, azimuth, elevation)
        self.camera_name = camera_name
        self.sim_camera = SimCamera(self.model, self.data)
        self.grasp_state = "idle"  # idle -> detect -> approach -> grasp -> lift -> done
        self.target_pos = None
        self.step_counter = 0

        # Panda gripper control
        self.gripper_open_val = 255
        self.gripper_close_val = 0

        # EE body
        self.ee_body_name = "hand"

        # Pre-grasp height offset
        self.approach_height = 0.15
        self.grasp_height = 0.02

    def runBefore(self):
        """Initialize: go to home position."""
        if self.model.nkey > 0:
            home = self.model.key_qpos[0]
            for i in range(min(len(home), self.model.nq)):
                self.data.qpos[i] = home[i]

        # Open gripper
        if self.model.nu > 7:
            self.data.ctrl[7] = self.gripper_open_val

        self.grasp_state = "detect"
        self.step_counter = 0

    def detect_object(self, rgb_image=None):
        """Detect the target object position from a camera image.

        Basic implementation: uses color segmentation to find a red object.
        Override this method for more sophisticated detection.

        Args:
            rgb_image: RGB image (H, W, 3). If None, uses default position.

        Returns:
            Estimated 3D position [x, y, z] in world frame, or None if not detected.
        """
        if rgb_image is None or not HAS_CV2:
            # Fallback: scan for non-floor geoms at a reachable position
            return self._detect_from_simulation()

        # Simple color-based detection (red objects)
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        # Red color range in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Largest contour centroid
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] < 1:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Map pixel to approximate world coordinates
        # This is a rough estimate; a proper implementation would use
        # camera intrinsics and depth to do proper projection.
        h, w = rgb_image.shape[:2]
        x_world = 0.3 + (cx / w - 0.5) * 0.6
        y_world = (cy / h - 0.5) * 0.6
        z_world = self.grasp_height

        return np.array([x_world, y_world, z_world])

    def _detect_from_simulation(self):
        """Detect graspable objects directly from simulation state."""
        workspace = {'x': [0.2, 0.7], 'y': [-0.4, 0.4], 'z': [0.0, 0.3]}
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name not in ["world", "link0", "link1", "link2", "link3",
                                      "link4", "link5", "link6", "link7",
                                      "hand", "left_finger", "right_finger"]:
                pos = self.data.body(i).xpos.copy()
                if (workspace['x'][0] < pos[0] < workspace['x'][1] and
                    workspace['y'][0] < pos[1] < workspace['y'][1] and
                    workspace['z'][0] < pos[2] < workspace['z'][1]):
                    return pos
        return None

    def _get_ee_pos(self):
        """Get current end-effector position."""
        ee_id = self.get_body_id(self.ee_body_name)
        return self.data.body(ee_id).xpos.copy()

    def runFunc(self):
        """Execute the grasp state machine."""
        self.step_counter += 1

        if self.grasp_state == "detect":
            # Try to capture image and detect
            rgb = None
            if self.camera_name:
                try:
                    rgb = self.sim_camera.capture_fixed(self.camera_name)
                except Exception:
                    pass
            self.target_pos = self.detect_object(rgb)

            if self.target_pos is not None:
                print(f"[Grasp] Object detected at: {self.target_pos}")
                self.grasp_state = "approach"
                self.approach_target = self.target_pos.copy()
                self.approach_target[2] += self.approach_height
            else:
                if self.step_counter % 100 == 0:
                    print("[Grasp] No object detected, scanning...")

        elif self.grasp_state == "approach":
            # Move EE above the object
            ee_pos = self._get_ee_pos()
            error = self.approach_target - ee_pos
            if np.linalg.norm(error) < 0.02:
                self.grasp_state = "descend"
                self.descend_target = self.target_pos.copy()
                self.descend_target[2] = self.grasp_height
                print("[Grasp] Approaching complete, descending...")
            else:
                # Simple proportional control on EE position
                # In practice, use IK + trajectory
                vel = 0.5 * error
                self.data.ctrl[:3] += vel[:3] * self.model.opt.timestep

        elif self.grasp_state == "descend":
            ee_pos = self._get_ee_pos()
            error = self.descend_target - ee_pos
            if np.linalg.norm(error) < 0.02:
                self.grasp_state = "close_gripper"
                print("[Grasp] Closing gripper...")
            else:
                vel = 0.3 * error
                self.data.ctrl[:3] += vel[:3] * self.model.opt.timestep

        elif self.grasp_state == "close_gripper":
            if self.model.nu > 7:
                self.data.ctrl[7] = self.gripper_close_val
            self.close_counter = getattr(self, 'close_counter', 0) + 1
            if self.close_counter > 50:
                self.grasp_state = "lift"
                self.close_counter = 0
                print("[Grasp] Lifting...")

        elif self.grasp_state == "lift":
            ee_pos = self._get_ee_pos()
            lift_target = self.target_pos.copy()
            lift_target[2] = 0.4
            error = lift_target - ee_pos
            if np.linalg.norm(error) < 0.02:
                self.grasp_state = "done"
                print("[Grasp] Grasp complete!")
            else:
                vel = 0.3 * error
                self.data.ctrl[:3] += vel[:3] * self.model.opt.timestep

        elif self.grasp_state == "done":
            pass  # Hold position
