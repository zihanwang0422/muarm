import gymnasium as gym
import panda_mujoco_gym
from stable_baselines3 import SAC, DDPG
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

def train_environment(env_id="FrankaPickAndPlaceSparse-v0", total_timesteps=1_000_000):
    print(f"================ 初始化环境: {env_id} ================")
    # 1. 创建环境（训练时一般不需要 render_mode="human" 以加快训练速度）
    env = gym.make(env_id)

    # 2. 定义 RL 模型配置
    # 由于环境返回 dict 类型的 observation，所以必须使用 "MultiInputPolicy"
    # 同时引入 HerReplayBuffer 以处理 Sparse Reward (稀疏奖励) 的问题
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,              # 打印训练日志
        buffer_size=1_000_000,  # 经验回放池大小
        batch_size=256,         # 批次大小
        learning_rate=1e-3,     # 学习率
        gamma=0.95,             # 折扣因子
        device="cuda"           # 自动选择 CPU/GPU
    )

    # 3. 开始训练
    print("开始训练，请耐心等待...")
    model.learn(total_timesteps=total_timesteps)

    # 4. 保存模型
    model_name = f"sac_{env_id.lower()}"
    model.save(model_name)
    print(f"模型已保存为: {model_name}.zip")

    env.close()

if __name__ == "__main__":
    # 你可以把以下名称替换为你想训练的任务之一：
    # "FrankaPushSparse-v0"
    # "FrankaSlideSparse-v0"
    # "FrankaPickAndPlaceSparse-v0"
    
    target_env = "FrankaPickAndPlaceSparse-v0" 
    train_environment(env_id=target_env, total_timesteps=10000)