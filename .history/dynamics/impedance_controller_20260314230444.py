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
    # --- 关节空间阻抗控制参数 ---
    # 分别对应 7 个关节的刚度和阻尼
    # 前四个关节（肩部和肘部）惯量大，需要较大的 P 和 D
    # 后三个关节（腕部）惯量小，P 和 D 要适当减小以防高频震荡
    KP_J = np.array([250.0, 250.0, 250.0, 250.0, 80.0, 50.0, 50.0])
    KD_J = np.array([30.0,  30.0,  30.0,  30.0,  10.0, 5.0,  5.0])

    # 阈值设置
    DRAG_DETECT_F  = 1.5   # 检测拖拽力 (N)
    RETURN_TRIGGER = 0.03  # 触发回弹的关节最大偏差 (rad)
    RETURN_DONE    = 0.01  # 认为已回到原位的关节最大偏差 (rad)

    def runBefore(self):
        # 初始化位置
        home = self.model.key_qpos[0].copy()
        self.data.qpos[:] = home
        mujoco.mj_forward(self.model, self.data)

        # 运动学与雅可比计算工具（此 Demo 现采用纯关节空间，不再强依赖雅可比）
        self.kin = PandaKinematics(ee_frame="hand")
        self.kin.build_from_mjcf(PANDA)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # 记录起始平衡关节状态
        self.eq_q = self.data.qpos[:7].copy()
        self.eq_pos = self.data.body(self.ee_id).xpos.copy()

        self._phase = "float"
        self._was_dragging = False

        print("\n" + "="*50)
        print(" Franka 关节空间阻抗控制 Demo")
        print(" 1. 双击机械臂任意连杆")
        print(" 2. 按住 Ctrl + 鼠标左键拖拽")
        print(" 3. 松开鼠标 -> 机械臂所有关节自动回弹至初始角度")
        print("="*50 + "\n")

    def runFunc(self):
        # 1. 获取状态
        q7 = self.data.qpos[:7].copy()
        v7 = self.data.qvel[:7].copy()

        # 使用 MuJoCo 自带的偏置力（包含重力、离心力等）
        g_tau = self.data.qfrc_bias[:7].copy()

        # 获取当前末端位姿（仅作可视化）
        ee_pos = self.data.body(self.ee_id).xpos.copy()
        
        # 计算关节角度偏差
        e_q = self.eq_q - q7
        # 取最大关节偏差用于状态机判断
        dist = np.max(np.abs(e_q)) 
        
        # 2. 检测拖拽交互 (检测整个机身上的外力)
        # 检查所有 Body 上的受外力大小（截取平动的三轴力）
        f_ext_all = np.linalg.norm(self.data.xfrc_applied[:, :3], axis=1)
        is_dragging = np.any(f_ext_all > self.DRAG_DETECT_F)

        # 3. 控制状态机
        if self._phase == "float":
            # 漂浮模式：仅重力补偿
            tau_imp = np.zeros(7)
            if self._was_dragging and not is_dragging and dist > self.RETURN_TRIGGER:
                self._phase = "return"
                print(f" [Return] 最大关节偏移 {dist:.3f} rad -> 开始回弹")
        else:
            # 回弹模式：关节空间阻抗 (PD 控制)
            # --- 关节空间虚拟力矩 tau = KP * e_q - KD * dot(q) ---
            tau_imp = self.KP_J * e_q - self.KD_J * v7

            # Debug: 打印关节级误差与力矩
            if dist > 0.05: # 有明显误差时才打印
                print(f"[Debug] e_q: {np.round(e_q, 3)}")
                print(f"[Debug] tau_imp: {np.round(tau_imp, 2)}")
                print(f"[Debug] v7: {np.round(v7, 3)}\n")

            if dist < self.RETURN_DONE:
                self._phase = "float"
                print(" [Float] 所有关节已回到原位")

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