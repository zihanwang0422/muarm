import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import mujoco
from src.mujoco_viewer import MuJoCoViewer
from kinematic.panda_kinematics import PandaKinematics
from utils.transform import quat2rotmat
import pinocchio as pin

SCENE = "models/franka_emika_panda/scene_torque.xml"
PANDA = "models/franka_emika_panda/panda.xml"

class FrankaImpedanceDemo(MuJoCoViewer):
    # --- 阻抗控制参数 ---
    # 增加 KP 使回弹更有力，增加 KD 防止震荡
    KP = np.array([600.0, 600.0, 600.0, 30.0, 30.0, 30.0])
    KD = np.array([50.0, 50.0, 50.0, 2.0, 2.0, 2.0])

    # 阈值设置
    DRAG_DETECT_F  = 1.5   # 检测拖拽力
    RETURN_TRIGGER = 0.05  # 触发回弹的最小位移 (5cm)
    RETURN_DONE    = 0.01  # 完成回弹的阈值 (1cm)

    def runBefore(self):
        # 初始化位置
        home = self.model.key_qpos[0].copy()
        self.data.qpos[:] = home
        mujoco.mj_forward(self.model, self.data)

        # 运动学与雅可比计算工具
        self.kin = PandaKinematics(ee_frame="hand")
        self.kin.build_from_mjcf(PANDA)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # 记录起始平衡位姿
        self.eq_pos = self.data.body(self.ee_id).xpos.copy()
        self.eq_R   = quat2rotmat(self.data.body(self.ee_id).xquat.copy())

        self._phase = "float"
        self._was_dragging = False

        print("\n" + "="*50)
        print(" Franka 任务空间阻抗控制 Demo")
        print(" 1. 双击机械臂末端 (hand)")
        print(" 2. 按住 Ctrl + 鼠标左键拖拽")
        print(" 3. 松开鼠标 -> 机械臂自动回弹")
        print("="*50 + "\n")

    def runFunc(self):
        # 1. 获取状态
        q7 = self.data.qpos[:7].copy()
        v7 = self.data.qvel[:7].copy()

        # 使用 MuJoCo 自带的偏置力（包含重力、离心力等），这是最准的
        g_tau = self.data.qfrc_bias[:7].copy()

        # 获取当前末端位姿
        ee_pos = self.data.body(self.ee_id).xpos.copy()
        ee_quat = self.data.body(self.ee_id).xquat.copy()
        R_cur = quat2rotmat(ee_quat)
        
        # 计算雅可比并只取前7列
        J_full = self.kin.jacobian(q7)
        J = J_full[:6, :7] 
        
        # 末端速度 v = J * qdot
        ee_vel = J @ v7
        dist = np.linalg.norm(ee_pos - self.eq_pos)

        # 2. 检测拖拽交互
        f_ext = np.linalg.norm(self.data.xfrc_applied[self.ee_id])
        is_dragging = f_ext > self.DRAG_DETECT_F

        # 3. 控制状态机
        if self._phase == "float":
            # 漂浮模式：仅重力补偿
            tau_imp = np.zeros(7)
            if self._was_dragging and not is_dragging and dist > self.RETURN_TRIGGER:
                self._phase = "return"
                print(f" [Return] 偏移 {dist*100:.1f}cm -> 开始回弹")
        else:
            # 回弹模式：阻抗控制
            # --- 位置误差 (Target - Current) ---
            e_pos = self.eq_pos - ee_pos
            
            # --- 姿态误差 (基于旋转矩阵) ---
            e_rot = 0.5 * (
                np.cross(R_cur[:, 0], self.eq_R[:, 0]) +
                np.cross(R_cur[:, 1], self.eq_R[:, 1]) +
                np.cross(R_cur[:, 2], self.eq_R[:, 2])
            )
            
            error = np.concatenate([e_pos, e_rot])
            
            # --- 计算笛卡尔空间虚拟力 F = K*e - D*v ---
            F_cartesian = self.KP * error - self.KD * ee_vel
            
            # --- 映射到关节力矩 tau = J^T * F ---
            tau_imp = J.T @ F_cartesian

            if dist < self.RETURN_DONE:
                self._phase = "float"
                print(" [Float] 已回到原位")

        self._was_dragging = is_dragging

        # 4. 最终力矩 = 阻抗力矩 + 重力补偿
        tau_final = tau_imp + g_tau
        
        # 施加到执行器并限幅
        self.data.ctrl[:7] = np.clip(tau_final, -80, 80)

        # 5. 可视化
        self.handle.user_scn.ngeom = 0
        self.add_visual_geom(
            [self.eq_pos, ee_pos],
            ["sphere", "sphere"],
            [[0.03], [0.02]],
            [[1, 0.8, 0, 0.8], [0, 1, 0, 0.8]]
        )
        if self._phase == "return":
            self.add_visual_lines([(ee_pos, self.eq_pos)], rgba=[1, 0, 0, 0.8])

if __name__ == "__main__":
    FrankaImpedanceDemo(SCENE, distance=2.5, azimuth=-45, elevation=-25).run_loop()