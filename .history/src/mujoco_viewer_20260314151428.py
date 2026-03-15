"""MuJoCo viewer base class with open APIs for simulation control.

Provides a CustomViewer/MuJoCoViewer abstraction that can be subclassed
for different simulation scenarios. External files can use the API to control
simulation, query states, and render visualizations.
"""
import time
import numpy as np
import mujoco
import mujoco.viewer
from xml.etree import ElementTree as ET

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.transform import euler2quat, quat2euler


class MuJoCoViewer:
    """Base MuJoCo viewer with open API for simulation control.

    Subclass this and override ``runBefore()`` and ``runFunc()`` to implement
    custom simulation logic. Call ``run_loop()`` to start the simulation.

    Example::

        class MyEnv(MuJoCoViewer):
            def runBefore(self):
                self.target = np.array([0.5, 0.0, 0.3])

            def runFunc(self):
                self.data.ctrl[:7] = some_controller(self.data, self.target)

        env = MyEnv("models/franka_emika_panda/scene.xml")
        env.run_loop()
    """

    def __init__(self, model_path, distance=3, azimuth=0, elevation=-30):
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.handle = None

    # ------------------------------------------------------------------ #
    #  Viewer lifecycle
    # ------------------------------------------------------------------ #
    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewport

    def set_timestep(self, timestep):
        self.model.opt.timestep = timestep

    # ------------------------------------------------------------------ #
    #  Main simulation loop
    # ------------------------------------------------------------------ #
    def run_loop(self):
        """Launch the viewer and run the simulation loop."""
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = self.distance
        self.handle.cam.azimuth = self.azimuth
        self.handle.cam.elevation = self.elevation
        self.runBefore()
        while self.is_running():
            mujoco.mj_forward(self.model, self.data)
            self.runFunc()
            mujoco.mj_step(self.model, self.data)
            self.sync()
            time.sleep(self.model.opt.timestep)

    def runBefore(self):
        """Called once before the simulation loop. Override in subclass."""
        pass

    def runFunc(self):
        """Called every simulation step. Override in subclass."""
        pass

    # ------------------------------------------------------------------ #
    #  Body / Geom query API
    # ------------------------------------------------------------------ #
    def get_body_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def get_body_names(self):
        names = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                names.append(name)
        return names

    def get_body_position(self, name):
        body_id = self.get_body_id(name)
        return self.data.body(body_id).xpos.copy()

    def get_body_quat(self, name):
        body_id = self.get_body_id(name)
        return self.data.body(body_id).xquat.copy()

    def get_body_pose(self, name):
        """Return [x, y, z, w, qx, qy, qz] for the named body."""
        pos = self.get_body_position(name)
        quat = self.get_body_quat(name)
        return np.concatenate([pos, quat])

    def get_body_pose_euler(self, name):
        """Return [x, y, z, roll, pitch, yaw] for the named body."""
        pos = self.get_body_position(name)
        quat = self.get_body_quat(name)
        euler = quat2euler(quat)
        return np.concatenate([pos, list(euler)])

    def get_geom_id(self, geom_name):
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name == geom_name:
                return i
        return -1

    def set_geom_position(self, geom_name, position):
        geom_id = self.get_geom_id(geom_name)
        if geom_id == -1:
            raise ValueError(f"Geom '{geom_name}' not found")
        self.model.geom_pos[geom_id] = position.copy()
        mujoco.mj_forward(self.model, self.data)

    def get_geom_position(self, geom_name):
        geom_id = self.get_geom_id(geom_name)
        if geom_id == -1:
            raise ValueError(f"Geom '{geom_name}' not found")
        return self.data.geom_xpos[geom_id].copy()

    # ------------------------------------------------------------------ #
    #  Mocap API
    # ------------------------------------------------------------------ #
    def set_mocap_position(self, name, position):
        body_id = self.get_body_id(name)
        mocap_id = self.model.body_mocapid[body_id]
        self.data.mocap_pos[mocap_id] = np.array(position)

    def set_mocap_quat(self, name, euler):
        body_id = self.get_body_id(name)
        mocap_id = self.model.body_mocapid[body_id]
        quat = euler2quat(*euler)
        self.data.mocap_quat[mocap_id] = np.array(quat)

    # ------------------------------------------------------------------ #
    #  Contact detection API
    # ------------------------------------------------------------------ #
    def get_contact_info(self):
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1_id = self.model.geom_bodyid[contact.geom1]
            body2_id = self.model.geom_bodyid[contact.geom2]
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            contacts.append({
                "geom1": contact.geom1, "geom2": contact.geom2,
                "pos": contact.pos.copy(),
                "body1_name": body1_name, "body2_name": body2_name,
            })
        return contacts

    # ------------------------------------------------------------------ #
    #  Visualization helpers
    # ------------------------------------------------------------------ #
    def add_visual_geom(self, positions, types, sizes, rgbas):
        """Add visual-only geometry to the viewer scene.

        Args:
            positions: (N, 3) array of positions
            types: list of type strings ("sphere", "box", "capsule", "cylinder")
            sizes: list of size arrays
            rgbas: (N, 4) array of RGBA colors
        """
        cur = self.handle.user_scn.ngeom
        self.handle.user_scn.ngeom = cur + len(positions)

        type_map = {
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        }

        for i in range(len(positions)):
            size = np.array(sizes[i], dtype=np.float64)
            while len(size) < 3:
                size = np.append(size, 0.0)
            mujoco.mjv_initGeom(
                self.handle.user_scn.geoms[i + cur],
                type=type_map.get(types[i], mujoco.mjtGeom.mjGEOM_SPHERE),
                size=size,
                pos=np.array(positions[i], dtype=np.float64),
                mat=np.eye(3).flatten(),
                rgba=np.array(rgbas[i], dtype=np.float32),
            )

    def add_obstacles(self, positions, types, sizes, rgbas):
        """Inject collision obstacles into the model by modifying the XML.

        Args:
            positions: list of [x, y, z] positions
            types: list of geometry type strings
            sizes: list of size arrays
            rgbas: list of [r, g, b, a] colors
        """
        tree = ET.parse(self.model_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("No <worldbody> found in model XML")

        for i in range(len(positions)):
            geom = ET.SubElement(worldbody, "geom")
            geom.set("name", f"obstacle_{i}")
            geom.set("type", types[i])
            geom.set("size", " ".join(f"{x:.3f}" for x in sizes[i]))
            geom.set("pos", f"{positions[i][0]:.3f} {positions[i][1]:.3f} {positions[i][2]:.3f}")
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.set("mass", "0.0")
            geom.set("rgba", " ".join(f"{c:.3f}" for c in rgbas[i]))

        new_path = self.model_path.replace(".xml", "_with_obstacles.xml")
        tree.write(new_path, encoding="utf-8", xml_declaration=True)
        self.model = mujoco.MjModel.from_xml_path(new_path)
        self.data = mujoco.MjData(self.model)

    # ------------------------------------------------------------------ #
    #  Joint state API
    # ------------------------------------------------------------------ #
    def get_joint_positions(self, n_joints=None):
        n = n_joints or self.model.nq
        return self.data.qpos[:n].copy()

    def get_joint_velocities(self, n_joints=None):
        n = n_joints or self.model.nv
        return self.data.qvel[:n].copy()

    def set_control(self, ctrl):
        """Set the control input. ``ctrl`` may be shorter than ``model.nu``."""
        n = min(len(ctrl), self.model.nu)
        self.data.ctrl[:n] = ctrl[:n]
