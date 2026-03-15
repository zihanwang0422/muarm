"""Example: Panda admittance control simulation.

Demonstrates the admittance controller responding to simulated external forces.
The robot maintains its position under external perturbation using a
virtual mass-spring-damper model.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src import MuJoCoViewer
from kinematic import PandaKinematics
from control import AdmittanceController
from utils import transform2mat

SCENE_XML = "models/franka_emika_panda/scene.xml"
ARM_XML = "models/franka_emika_panda/panda.xml"


class AdmittanceDemo(MuJoCoViewer):
    def __init__(self):
        super().__init__(SCENE_XML, distance=3, azimuth=-45, elevation=-30)

    def runBefore(self):
        self.set_timestep(0.001)

        # Initialize at home position
        if self.model.nkey > 0:
            home = self.model.key_qpos[0]
            self.data.ctrl[:7] = home[:7]

        # Kinematics for IK
        self.kin = PandaKinematics(ee_frame="link7")
        self.kin.build_from_mjcf(ARM_XML)

        # Admittance controller
        self.admittance = AdmittanceController(M=10.0, B=50.0, K=100.0)

        self.init_counter = 100
        self.last_dof = self.data.qpos.copy()
        self.ee_body = "hand"

    def runFunc(self):
        if self.init_counter > 0:
            # Let the robot settle at home position
            self.init_counter -= 1
            if self.model.nkey > 0:
                self.data.ctrl[:7] = self.model.key_qpos[0][:7]
            ee_pose = self.get_body_pose_euler(self.ee_body)
            self.admittance.set_reference(ee_pose)
            self.last_ee_pose = ee_pose
            return

        # Get current EE state
        ee_pose = self.get_body_pose_euler(self.ee_body)
        ee_vel = (ee_pose - self.last_ee_pose) / self.model.opt.timestep
        self.last_ee_pose = ee_pose.copy()

        # Simulated external force: push down along z
        F_ext = np.zeros(6)
        F_ext[2] = -5.0  # 5N downward force

        # Admittance update
        desired_pose = self.admittance.update(F_ext, ee_pose, ee_vel, self.model.opt.timestep)

        # Solve IK for desired pose
        tf = transform2mat(*desired_pose[:6])
        dof, info = self.kin.ik(tf, q_init=self.last_dof)

        if info["success"]:
            self.last_dof = dof
            self.data.qpos[:7] = dof[:7]


if __name__ == "__main__":
    demo = AdmittanceDemo()
    demo.run_loop()
