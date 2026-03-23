import time
import gymnasium as gym
import panda_mujoco_gym
from stable_baselines3 import SAC

def test_environment(env_id="FrankaPickAndPlaceSparse-v0"):
    print(f"================ 加载环境: {env_id} ================")
    
    # 1. 创建环境，必须设置 render_mode="human" 才能看到 MuJoCo 可视化窗口
    env = gym.make(env_id, render_mode="human")
    
    # 2. 找到对应的模型权重文件
    # 与训练时保存时的模型名称对应
    model_name = f"sac_{env_id.lower()}"
    model_path = f"{model_name}.zip"
    
    # 3. 加载训练好的模型
    try:
        print(f"正在加载模型: {model_path}")
        model = SAC.load(model_path, env=env)
        print("模型加载成功！")
    except FileNotFoundError:
        print(f"找不到模型文件 {model_path}，请确认是否已经运行了刚刚的训练脚本。")
        env.close()
        return

    # 4. 重置环境并获取初始观测值
    observation, info = env.reset()

    # 5. 循环执行仿真
    for step in range(1000):
        # 让模型根据当前的观测预测下一步动作
        # deterministic=True 代表使用确定性策略（测试时为了追求最佳表现，通常关闭探索）
        action, _states = model.predict(observation, deterministic=True)
        
        # 将动作输入到环境中
        observation, reward, terminated, truncated, info = env.step(action)

        # 如果任务完成或超时（回合结束），重置环境
        if terminated or truncated:
            print(f"回合结束，状态: 成功={info.get('is_success', False)}，正在重置...")
            observation, info = env.reset()
            # 在下次开始前稍作停顿，方便观察
            time.sleep(0.1)

    # 关闭环境
    env.close()

if __name__ == "__main__":
    # 需要和刚才刚才训练的任务名称一致
    # "FrankaPushSparse-v0"
    # "FrankaSlideSparse-v0"
    # "FrankaPickAndPlaceSparse-v0"
    
    target_env = "FrankaSlideSparse-v0"
    test_environment(env_id=target_env)