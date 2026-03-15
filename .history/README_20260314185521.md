# Manipulation MuJoCo

MuJoCo-based manipulation environment for the Franka Emika Panda robot.
Modular design: `src` / `kinematic` / `control` / `learning` / `grasp` / `utils`.

---
然后导纳控制admittance需要实现我在mujoco可视化界面使用鼠标拖拽末端执行器 然后实现柔顺的drag teaching 拖拽示教 
跟随
## Environment Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtualenv and install dependencies

```bash
cd manipulation_mujoco
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

> If `pin` (Pinocchio) fails to install, try:
> ```bash
> uv pip install cmeel-pinocchio
> uv pip install -r requirements.txt
> ```

### 3. Run a script

```bash
# With the venv activated:
python examples/demo_kinematics.py

# Or without activating via uv run:
uv run python examples/demo_kinematics.py
```

---

## Modules

### src — Core Simulation

Base viewer class (`MuJoCoViewer`), PID controller, MPC controller, and the
`KinematicsVisualizer` interface for FK/IK/Trajectory visualisation.

All other modules are built on top of `MuJoCoViewer`.

### kinematic — Kinematics & Trajectory

Panda FK/IK solver (Pinocchio, damped Jacobian) and trajectory generators
(cubic, linear, multi-waypoint, Cartesian arc).

Run the FK/IK/Trajectory demo:

```bash
python examples/demo_kinematics.py
```

Three stages open in sequence (close the viewer window to advance to the next):

1. **FK** — robot sweeps through four joint configurations; a blue sphere marks the EE.
2. **IK** — a red sphere appears at each Cartesian target; the robot solves IK and moves there.
3. **Trajectory** — robot follows a cubic multi-waypoint path; coloured spheres mark each waypoint.

Run only the trajectory demo:

```bash
python examples/demo_trajectory.py
```

### control — Robot Controllers

Joint-space and Cartesian impedance controller, and Cartesian admittance
controller (virtual M-B-K mass-spring-damper).

**Impedance demo** — a red target sphere orbits in a circle; the robot tracks it
using Cartesian impedance. A sinusoidal disturbance force is applied every few
seconds to show the restoring behaviour:

```bash
python examples/demo_impedance_control.py
```

**Admittance demo** — a periodic external force makes the EE "give way".
Three spheres show the equilibrium (yellow), the admittance-modified desired pose
(blue), and the live EE position (green):

```bash
python examples/demo_admittance_control.py
```

### learning — Reinforcement Learning

Gymnasium environments for Panda EE reaching and obstacle avoidance, plus PPO
training utilities (Stable-Baselines3).

**Train** a PPO agent (~5 M steps, saves to `assets/panda_ppo_reach.zip`):

```bash
python examples/demo_ppo_reach.py train
```

**Test** the trained agent — loads the saved model and opens a MuJoCo viewer.
The agent drives the arm to randomly sampled goal positions continuously:

```bash
python examples/demo_ppo_reach.py test
```

### grasp — Vision-based Grasping

`SimCamera` for RGB/depth capture from MJCF-defined cameras, and `GraspPipeline`
implementing a detect-approach-grasp-lift state machine.

To run the basic pipeline (object detection from the overhead camera):

```bash
python grasp/grasp_pipeline.py
```

### utils — Math Utilities

Quaternion, Euler, rotation-matrix conversions, homogeneous transform
helpers, and the damped pseudo-inverse. Imported across all other modules.
