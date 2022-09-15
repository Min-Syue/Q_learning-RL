from model.Value_Base import Q_learning
from model.funs import agent_walk_path
import gym
import numpy as np

if __name__ == '__main__':
    
    # 宣告Taxi的環境 並初始化
    env = gym.make("Taxi-v3")
    observation, info = env.reset()

    # 宣告Q-learning模型 並做訓練
    Q_model = Q_learning()
    q_table = Q_model.train_step(env=env)
    
    test_env = gym.make("Taxi-v3", render_mode="human")

    epoch, reward = agent_walk_path(test_env, q_table)

    print(f"Epochs:", epoch)
    print(f"Reward:", reward)