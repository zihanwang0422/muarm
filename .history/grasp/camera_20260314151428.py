"""Simulated camera for MuJoCo: capture RGB and depth images."""
import numpy as np
import mujoco

try:
    import glfw
    HAS_GLFW = True
except ImportError:
    HAS_GLFW = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class SimCamera:
    """Simulated camera that captures images from a MuJoCo scene.

    Supports both fixed cameras (defined in XML) and tracking cameras
    that follow a body.

    Example::

        cam = SimCamera(model, data)
        rgb = cam.capture_fixed("rgb_camera", width=640, height=480)
        depth = cam.capture_depth("rgb_camera", width=640, height=480)
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self._glfw_inited = False
        self._scene = None
        self._context = None
        self._camera = None

    def _init_offscreen(self, width, height):
        """Initialize GLFW offscreen context for rendering."""
        if self._glfw_inited:
            return
        if not HAS_GLFW:
            raise RuntimeError("glfw is required for camera capture: pip install glfw")
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self._window = glfw.create_window(width, height, "Offscreen", None, None)
        glfw.make_context_current(self._window)

        self._scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self._context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context)
        self._camera = mujoco.MjvCamera()
        self._glfw_inited = True

    def _render(self, width, height):
        """Render the scene and return raw RGB buffer."""
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model, self.data, mujoco.MjvOption(),
            mujoco.MjvPerturb(), self._camera,
            mujoco.mjtCatBit.mjCAT_ALL, self._scene,
        )
        mujoco.mjr_render(viewport, self._scene, self._context)

        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth_buf = np.zeros((height, width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth_buf, viewport, self._context)

        # Flip vertically (OpenGL origin is bottom-left)
        rgb = np.flipud(rgb)
        depth_buf = np.flipud(depth_buf)
        return rgb, depth_buf

    def capture_fixed(self, camera_name, width=640, height=480):
        """Capture RGB image from a fixed camera defined in the MJCF.

        Args:
            camera_name: Name of the camera in the model XML
            width: Image width
            height: Image height

        Returns:
            RGB image as (H, W, 3) numpy array (uint8)
        """
        self._init_offscreen(width, height)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._camera.fixedcamid = cam_id
        rgb, _ = self._render(width, height)
        return rgb

    def capture_tracking(self, body_name, distance=0.5, azimuth=0, elevation=-30,
                         width=640, height=480):
        """Capture RGB image from a camera tracking a body.

        Args:
            body_name: Name of the body to track
            distance: Camera distance from body
            azimuth: Camera azimuth angle (degrees)
            elevation: Camera elevation angle (degrees)
            width: Image width
            height: Image height

        Returns:
            RGB image as (H, W, 3) numpy array (uint8)
        """
        self._init_offscreen(width, height)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._camera.trackbodyid = body_id
        self._camera.distance = distance
        self._camera.azimuth = azimuth
        self._camera.elevation = elevation
        rgb, _ = self._render(width, height)
        return rgb

    def capture_depth(self, camera_name, width=640, height=480):
        """Capture depth image from a fixed camera.

        Args:
            camera_name: Name of the camera
            width: Image width
            height: Image height

        Returns:
            Depth image as (H, W) float32 array. Values in [0, 1], where
            0 = near clipping plane, 1 = far clipping plane.
        """
        self._init_offscreen(width, height)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._camera.fixedcamid = cam_id
        _, depth = self._render(width, height)
        return depth

    def depth_to_meters(self, depth_buf):
        """Convert MuJoCo normalized depth buffer to meters.

        Args:
            depth_buf: Normalized depth buffer from capture_depth()

        Returns:
            Depth in meters (H, W) float32 array
        """
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        return near / (1.0 - depth_buf * (1.0 - near / far))

    def rgb_to_bgr(self, rgb):
        """Convert RGB to BGR for OpenCV display."""
        if HAS_CV2:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return rgb[:, :, ::-1].copy()
