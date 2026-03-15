# SO-ARM100 MuJoCo Model-Based Manipulation

> **一站式项目**：基于 MuJoCo 物理引擎 + 官方 TRS SO-ARM100 MJCF 模型，实现
> **正运动学 (FK)** · **逆运动学 (IK)** · **三次多项式轨迹规划** · **仿真执行** 的完整工作流。

![SO-ARM100](models/trs_so_arm100/so_arm100.png)

---

## 目录

1. [项目概览](#1-项目概览)
2. [环境搭建](#2-环境搭建)
3. [项目结构](#3-项目结构)
4. [机械臂模型说明](#4-机械臂模型说明)
5. [模块详解](#5-模块详解)
   - [5.1 正运动学 (FK)](#51-正运动学-fk)
   - [5.2 逆运动学 (IK)](#52-逆运动学-ik)
   - [5.3 轨迹规划](#53-轨迹规划)
   - [5.4 MuJoCo 仿真器](#54-mujoco-仿真器)
   - [5.5 数据生成器](#55-数据生成器)
6. [使用方法](#6-使用方法)
7. [Demo 说明](#7-demo-说明)
8. [从 ROS/Unity 迁移说明](#8-从-rosunity-迁移说明)
9. [常见问题](#9-常见问题)

---

## 1. 项目概览

本项目将之前基于 **ROS 2 + Unity CUTeR Simulator** 的 SO-ARM100 机械臂控制与运动规划代码，
迁移至 **MuJoCo** 物理引擎。核心功能包括：

| 功能 | 实现方式 |
|------|----------|
| **正运动学** | 自定义 FK（Rodrigues 公式 + body chain），可与 MuJoCo 内建 FK 交叉验证 |
| **逆运动学** | 阻尼最小二乘法 (DLS/LM)；可选：预训练神经网络 |
| **轨迹规划** | 三次多项式插值（关节空间 / 笛卡尔空间 / 多路点链式） |
| **仿真执行** | MuJoCo 位置执行器（kp=50），支持被动 viewer 和阻塞式 viewer |
| **数据生成** | 自动生成 IK 训练数据集（FK + 扰动），输出 CSV |

---

## 2. 环境搭建

### 2.1 创建 Conda 环境

```bash
conda create -n soarm_mujoco python=3.10 -y
conda activate soarm_mujoco
```

### 2.2 安装依赖

```bash
cd mujoco_sim
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
mujoco>=3.0.0
numpy>=1.23
pandas>=1.5
matplotlib>=3.5
scikit-learn>=1.1
scipy>=1.9
# Optional — only needed for NN-based IK
# tensorflow>=2.10
```

### 2.3 验证安装

```bash
conda activate soarm_mujoco
python -c "import mujoco; print('MuJoCo', mujoco.__version__)"
```

---

## 3. 项目结构

```
mujoco_sim/
├── main.py                          # 主入口 (demo runner)
├── requirements.txt                 # Python 依赖
├── README.md                        # 本文档
├── __init__.py
│
├── models/
│   └── trs_so_arm100/               # 官方 TRS MJCF 模型 (勿修改)
│       ├── so_arm100.xml            #   机械臂本体定义
│       ├── scene.xml                #   原始场景
│       ├── scene_sim.xml            #   增强场景 (传感器+目标体)
│       └── assets/                  #   18 个 STL 网格文件
│
├── kinematics/
│   ├── __init__.py
│   ├── forward_kinematics.py        # 正运动学
│   └── inverse_kinematics.py        # 逆运动学 (数值 + NN)
│
├── trajectory.py                    # 三次多项式轨迹规划
├── simulator.py                     # MuJoCo 仿真器包装
│
└── utils/
    ├── __init__.py
    └── data_generator.py            # IK 训练数据集生成
```

---

## 4. 机械臂模型说明

### 4.1 运动链

使用 **官方 TRS MuJoCo MJCF** 模型（`so_arm100.xml`），运动链如下：

```
World Base
  └─ Rotation_Pitch         ← J0: Rotation    (Y-axis)
       └─ Upper_Arm         ← J1: Pitch       (X-axis)
            └─ Lower_Arm    ← J2: Elbow       (X-axis)
                 └─ Wrist_Pitch_Roll ← J3: Wrist_Pitch (X-axis)
                      └─ Fixed_Jaw  ← J4: Wrist_Roll  (Y-axis)
                           └─ Jaw   ← J5: Jaw gripper (Z-axis)
```

- **5 个手臂关节** (`J0`–`J4`) 用于定位和姿态控制
- **1 个夹爪关节** (`J5 Jaw`) 用于抓取

### 4.2 关节参数

| 关节 | 轴 | 范围 (rad) | 范围 (deg) |
|------|----|-----------|-----------|
| `Rotation`     | Y `[0,1,0]` | [−1.92, 1.92]  | ±110° |
| `Pitch`        | X `[1,0,0]` | [−3.32, 0.174] | −190°~10° |
| `Elbow`        | X `[1,0,0]` | [−0.174, 3.14] | −10°~180° |
| `Wrist_Pitch`  | X `[1,0,0]` | [−1.66, 1.66]  | ±95° |
| `Wrist_Roll`   | Y `[0,1,0]` | [−2.79, 2.79]  | ±160° |
| `Jaw`          | Z `[0,0,1]` | [−0.174, 1.75] | −10°~100° |

### 4.3 Body-to-Body 变换

每个 body 之间的固定偏移和四元数旋转（从 MJCF 提取）：

| 父 → 子 | 偏移 (m) | 四元数 (wxyz) |
|----------|---------|---------------|
| Base → Rotation_Pitch | `[0, -0.0452, 0.0165]` | `[0.707, 0.707, 0, 0]` |
| Rotation_Pitch → Upper_Arm | `[0, 0.1025, 0.0306]` | `[0.707, 0.707, 0, 0]` |
| Upper_Arm → Lower_Arm | `[0, 0.11257, 0.028]` | `[0.707, -0.707, 0, 0]` |
| Lower_Arm → Wrist_Pitch_Roll | `[0, 0.0052, 0.1349]` | `[0.707, -0.707, 0, 0]` |
| Wrist_Pitch_Roll → Fixed_Jaw | `[0, -0.0601, 0]` | `[0.707, 0, 0.707, 0]` |

### 4.4 关键帧

| 名称 | 关节角 (rad) |
|------|-------------|
| `home` | `[0, -1.57, 1.57, 1.57, -1.57, 0]` |
| `rest` | `[0, -3.32, 3.11, 1.18, 0, -0.174]` |

### 4.5 执行器

官方模型使用**位置执行器**：

```xml
<actuator>
  <position name="Rotation" joint="Rotation" kp="50" dampratio="1" forcerange="-3.5 3.5"/>
  ...
</actuator>
```

> **`ctrl = 目标关节角度 (rad)`**，而非力矩。这与 ROS 的 `JointTrajectoryController` 类似。

---

## 5. 模块详解

### 5.1 正运动学 (FK)

**文件**: `kinematics/forward_kinematics.py`

```python
from mujoco_sim.kinematics import ForwardKinematics

fk = ForwardKinematics()
pos, quat = fk.compute([0, -1.57, 1.57, 1.57, -1.57])  # home
# pos:  [x, y, z] in metres
# quat: [w, x, y, z]
```

#### 算法

1. 沿 body chain 逐级乘变换矩阵：

$$T_{i} = T_{i-1} \cdot \begin{bmatrix} R_{\text{body}} \cdot R_{\text{joint}}(\theta_i) & p_{\text{body}} \\ 0 & 1 \end{bmatrix}$$

2. 其中 $R_{\text{body}}$ 由 MJCF 中每个 body 的固定四元数决定
3. $R_{\text{joint}}(\theta_i)$ 由 Rodrigues 公式计算：

$$R(\hat{k}, \theta) = I + \sin\theta \cdot [k]_\times + (1-\cos\theta) \cdot [k]_\times^2$$

#### 方法

| 方法 | 说明 |
|------|------|
| `compute(angles)` | 5 关节角 → EE 位姿 `(pos, quat_wxyz)` |
| `compute_all_frames(angles)` | 返回所有 body 的世界坐标 `[(pos, rotmat), ...]` |
| `quat_to_rotmat(q)` | 四元数 → 3×3 旋转矩阵 |
| `rotmat_to_quat_wxyz(R)` | 旋转矩阵 → 四元数 |
| `axis_angle_to_rotmat(axis, angle)` | Rodrigues 公式 |

---

### 5.2 逆运动学 (IK)

**文件**: `kinematics/inverse_kinematics.py`

#### 5.2.1 数值 IK (Damped Least-Squares)

```python
from mujoco_sim.kinematics import InverseKinematics

ik = InverseKinematics()
angles = ik.numerical(
    target_pos=np.array([0.1, -0.05, 0.2]),
    target_quat_wxyz=np.array([1, 0, 0, 0]),
    max_iter=300, tol=1e-4, damping=0.05,
    pos_weight=1.0, ori_weight=0.3,
)
```

算法核心：

$$\Delta\theta = J_w^T (J_w J_w^T + \lambda I)^{-1} \cdot e_w$$

其中 $J_w = WJ$ 为加权雅可比，$e_w = We$ 包含位置和姿态误差，
$\lambda$ 为阻尼系数（避免奇异点）。

雅可比矩阵通过**有限差分**计算（6×5）：
- 行 0–2: 位置偏导
- 行 3–5: 轴角姿态偏导

每步迭代后将关节角 clamp 到合法范围。

#### 5.2.2 神经网络 IK (可选)

```python
ik.load_nn_model("model.keras", "scaler.pkl")
angles = ik.neural_network(target_pos, target_quat, reference_angles)
```

- 输入: 12 维 = `[参考关节角(5), 目标四元数(4), 目标位置(3)]`
- 输出: 5 个目标关节角
- 需要 `tensorflow` 和预训练模型

---

### 5.3 轨迹规划

**文件**: `trajectory.py`

#### 三次多项式插值

给定边界条件 $(q_0, \dot{q}_0, q_f, \dot{q}_f, T)$：

$$q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3$$

$$\begin{cases}
a_0 = q_0 \\
a_1 = \dot{q}_0 \\
a_2 = \frac{3(q_f - q_0)}{T^2} - \frac{2\dot{q}_0}{T} - \frac{\dot{q}_f}{T} \\
a_3 = \frac{2(q_0 - q_f)}{T^3} + \frac{\dot{q}_0 + \dot{q}_f}{T^2}
\end{cases}$$

#### 方法

| 方法 | 说明 |
|------|------|
| `joint_trajectory(start, end, duration, freq)` | 关节空间三次插值 |
| `task_trajectory(start_pos, end_pos, start_euler, end_euler, ...)` | 笛卡尔空间三次插值 (含欧拉角解缠绕) |
| `task_to_joint(task_traj, initial_angles)` | 笛卡尔 → 关节 (逐点 IK) |
| `multi_waypoint(waypoints, durations, freq)` | 多路点链式三次轨迹 |

```python
from mujoco_sim.trajectory import TrajectoryGenerator

tg = TrajectoryGenerator()

# 关节空间
traj = tg.joint_trajectory(
    start=[0, -1.57, 1.57, 1.57, -1.57],
    end=[0.5, -1.0, 1.2, 0.8, -0.5],
    duration=3.0, frequency=50,
)

# 多路点
traj = tg.multi_waypoint(
    waypoints=[wp1, wp2, wp3, wp4],
    durations=[2.0, 1.5, 2.0],
    frequency=50,
)
```

---

### 5.4 MuJoCo 仿真器

**文件**: `simulator.py`

```python
from mujoco_sim.simulator import MuJoCoSimulator

sim = MuJoCoSimulator(render=True)
sim.reset_to_keyframe("home")
sim.set_arm_target([0.3, -1.0, 1.2, 0.5, -0.2])
sim.step(200)  # 模拟 200 步
print(sim.get_ee_position())
```

#### 特性

- 自动加载 `scene_sim.xml`（增强场景，含传感器和目标物体）
- 位置执行器：`ctrl[i] = 目标角度`（不是力矩！）
- 14 个传感器：6 关节位置 + 6 关节速度 + EE 位置 + EE 四元数
- 被动 viewer (实时渲染) 和阻塞式 viewer (交互式)

#### 主要方法

| 方法 | 说明 |
|------|------|
| `reset(arm_angles, jaw)` | 重置仿真状态 |
| `reset_to_keyframe("home")` | 重置到关键帧 |
| `set_arm_target(angles)` | 设置 5 个手臂执行器目标 |
| `set_jaw(angle)` | 设置夹爪目标 |
| `step(n)` | 推进 n 个仿真步 |
| `execute_trajectory(traj, freq)` | 执行整条轨迹 + viewer 同步 |
| `get_arm_positions()` | 读取 5 个手臂关节传感器 |
| `get_ee_position()` | 读取 EE 位置 (m) |
| `get_ee_quaternion()` | 读取 EE 四元数 (wxyz) |
| `get_state()` | 获取完整状态字典 |
| `launch_viewer()` | 阻塞式交互 viewer |

---

### 5.5 数据生成器

**文件**: `utils/data_generator.py`

为 NN-IK 生成训练数据：

```python
from mujoco_sim.utils.data_generator import generate_ik_dataset

df = generate_ik_dataset(num_samples=5000, output_csv="ik_dataset.csv")
```

- 在关节空间随机采样参考角度
- 对每个样本进行多次小扰动
- 用 FK 计算对应的 EE 位姿
- 输出 12 输入 + 5 目标的 CSV

---

## 6. 使用方法

### 6.1 运行所有 Demo (headless)

```bash
conda activate soarm_mujoco
cd /path/to/SO-ARM100_IK_solver
python -m mujoco_sim.main
```

### 6.2 运行单个 Demo + 可视化

```bash
python -m mujoco_sim.main --demo fk          # FK 验证
python -m mujoco_sim.main --demo ik --render  # IK + 3D viewer
python -m mujoco_sim.main --demo traj --render
python -m mujoco_sim.main --demo pick --render
python -m mujoco_sim.main --demo task --render
python -m mujoco_sim.main --demo view         # 交互式 viewer
```

### 6.3 在自己的代码中使用

```python
from mujoco_sim.kinematics import ForwardKinematics, InverseKinematics
from mujoco_sim.trajectory import TrajectoryGenerator
from mujoco_sim.simulator import MuJoCoSimulator

# FK
fk = ForwardKinematics()
pos, quat = fk.compute([0, -1.57, 1.57, 1.57, -1.57])

# IK
ik = InverseKinematics()
angles = ik.numerical(pos, quat)

# Trajectory
tg = TrajectoryGenerator()
traj = tg.joint_trajectory([0]*5, angles, duration=3.0, frequency=50)

# Simulate
sim = MuJoCoSimulator(render=True)
sim.reset_to_keyframe("home")
sim.execute_trajectory(traj, frequency=50)
sim.close()
```

### 6.4 生成 IK 数据集

```bash
python -m mujoco_sim.utils.data_generator
```

---

## 7. Demo 说明

| Demo | 功能 | 关键点 |
|------|------|--------|
| `fk` | 对比自定义 FK 与 MuJoCo 内建结果 | 在 Zero / Home / 随机构型下比较位置误差 |
| `ik` | 数值 IK 求解 + 圆trip 验证 | 先 FK 求目标，再 IK 反算，检查误差 |
| `traj` | 三次关节轨迹执行 | Home → 目标位姿，检查最大速度和终端误差 |
| `pick` | 8 路点模拟抓取 | Home→预取→取→提→运→放→回→Home |
| `task` | 笛卡尔轨迹 → IK | 笛卡尔插值后逐点 IK 转关节空间执行 |
| `view` | 交互式 3D viewer | 可拖拽、旋转、缩放模型 |

---

## 8. 从 ROS/Unity 迁移说明

### 8.1 之前的架构 (hw1–hw3)

```
Unity CUTeR Simulator ←──ROS 2──→ publisher.py / subscriber.py
                                        │
                                   SOARM101.py (FK/IK)
                                        │
                              cubic_trajectory_generation.py
```

### 8.2 现在的架构

```
MuJoCo (physics engine)
    │
    ├── simulator.py (wrapper)
    │
    ├── kinematics/
    │   ├── forward_kinematics.py
    │   └── inverse_kinematics.py
    │
    ├── trajectory.py
    │
    └── main.py (demo entry)
```

### 8.3 关键变化

| 项目 | 旧 (hw1–hw3) | 新 (mujoco_sim) |
|------|-------------|----------------|
| 仿真器 | Unity + ROS 2 topic | MuJoCo + Python API |
| 通信 | `JointState` ROS msg | 直接函数调用 |
| 关节轴 | Z, X, X, X, Y, X | Y, X, X, X, Y, Z（官方模型） |
| 自由度 | 6 DOF | 5 DOF arm + 1 gripper |
| 执行器 | 未知 / torque | 位置执行器 `ctrl=目标角度` |
| 坐标 | 自定义 DH | MJCF body chain + 四元数 |
| 单位 | 有时混用 mm/m | 统一 metres |
| IK | 几何解 + NN | 数值 DLS + NN |
| 渲染 | Unity 3D | MuJoCo viewer (OpenGL) |

---

## 9. 常见问题

### Q: 运行报错 `Model not found`

检查工作目录和模型路径。代码通过 `os.path.dirname(__file__)` 自动定位模型，
请确保从项目根目录运行 `python -m mujoco_sim.main`。

### Q: FK 和 MuJoCo 结果有微小偏差

正常。自定义 FK 使用浮点运算链式乘法，MuJoCo 内部使用其自己的变换方式。
偏差通常 < 0.1 mm。

### Q: 没有显示窗口 (headless)

- 确保安装了 OpenGL (如 `libglfw3`)
- 在 SSH 远程环境下需要 X11 forwarding 或 EGL 渲染
- 可用 `--render` 开启 viewer，默认 headless 运行

### Q: IK 不收敛

- 目标位姿可能超出工作空间
- 尝试提供更好的 `initial_angles`（接近目标的初始猜测）
- 增大 `max_iter` 或调小 `tol`
- 5-DOF 臂无法实现任意 6D 位姿

### Q: 如何使用神经网络 IK

1. 安装 `pip install tensorflow`
2. 生成训练数据：`python -m mujoco_sim.utils.data_generator`
3. 训练模型（参考 `src/hw3/ik_nn.py`）
4. 加载：`ik.load_nn_model("model.keras", "scaler.pkl")`

---

## 许可

模型文件遵循 TRS SO-ARM100 原始许可。代码部分仅用于学术目的。
