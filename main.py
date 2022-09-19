from model.Value_Base import Q_learning, Sarsa, DeepQ_learning
from model.funs import agent_walk_path_vedio
from keras.optimizers import Adam

import gym

if __name__ == '__main__':
    
    # 儲存影片的file name
    file_name = "videos_sarsa"

    # 宣告Taxi的環境 並初始化
    env = gym.make("Taxi-v3")
    observation, info = env.reset()

    '''
    # 宣告Q-learning模型 並做訓練
    Q_model = Q_learning()
    q_table = Q_model.train_step(env=env)
    '''
    '''
    # 宣告Sarsa模型 並做訓練
    Sarsa_model = Sarsa()
    q_table = Sarsa_model.train_step(env=env)

    test_env = gym.make("Taxi-v3", render_mode="rgb_array_list")

    epoch, reward = agent_walk_path_vedio(test_env, q_table, file_name)
    
    print(f"Epochs:", epoch)
    print(f"Reward:", reward)

    test_env.close()

    '''

    DQL = DeepQ_learning(enviroment=env, optimizer=Adam(10e-3))
    DQL.train_step(50000, 50, 32, 15, True)
    env.close()


    