"""PPO end-effector reaching — train then visualize in MuJoCo.

Usage:
    python examples/demo_ppo_reach.py train   # train PPO (~5 M steps, ~10 min)
    python examples/demo_ppo_reach.py test    # load saved model, run in MuJoCo

The saved model is stored at assets/panda_ppo_reach.zip.
During test mode a MuJoCo viewer opens and the agent drives the arm to
randomly sampled goal positions (green sphere = goal, red sphere = EE).

Run:
    python examples/demo_ppo_reach.py train
    python examples/demo_ppo_reach.py test
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

MODEL_PATH  = "assets/panda_ppo_reach"
SCENE       = "models/franka_emika_panda/scene.xml"
N_ENVS      = 8
TOTAL_STEPS = 5_000_000

# ── helpers ────────────────────────────────────────────────────────────────────

def do_train():
    from learning import PandaReachEnv, train_ppo
    os.makedirs("assets", exist_ok=True)
    train_ppo(
        env_class=PandaReachEnv,
        n_envs=N_ENVS,
        total_timesteps=TOTAL_STEPS,
        model_save_path=MODEL_PATH,
        env_kwargs={"scene_xml": SCENE},
    )
    print(f"Model saved → {MODEL_PATH}.zip")


def do_test():
    """Load the saved PPO model and visualise in MuJoCo."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        sys.exit("stable-baselines3 not installed: pip install stable-baselines3")

    if not os.path.exists(MODEL_PATH + ".zip"):
        sys.exit(f"No model found at {MODEL_PATH}.zip — run with 'train' first.")

    # Use PandaReachEnv with render_mode="human" so MuJoCo opens automatically
    from learning import PandaReachEnv
    env   = PandaReachEnv(render_mode="human", scene_xml=SCENE)
    model = PPO.load(MODEL_PATH, env=env)

    print("Running trained agent — close the MuJoCo window to stop.")
    ep = 0
    while True:
        obs, _ = env.reset()
        done   = False
        total  = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        ep += 1
        result = "SUCCESS" if info.get("is_success") else "FAIL"
        print(f"  Episode {ep:3d}  reward={total:7.2f}  {result}")

    env.close()

# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "train":
        do_train()
    elif mode == "test":
        do_test()
    else:
        print("Usage: python examples/demo_ppo_reach.py [train|test]")
