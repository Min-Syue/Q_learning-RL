import numpy as np

def agent_walk_path(env, q_table):
    
    # 宣告參數
    done = False
    epochs = 0
    total_reward = 0

    env.reset()

    while not done:

        # 現在環境的代號
        state = env.s
    
        # 現在最大q值的action
        action = np.argmax(q_table[state])

        state, reward, done, _, _ = env.step(action)

        epochs += 1
        total_reward += reward

    return epochs, total_reward