from tqdm import tqdm

import random
import numpy as np


class Q_learning:

    def __init__(self, alpha=0.1, gamma=0.6, epslion=0.1):

        # 設定參數
        self._alpha = alpha
        self._gamma = gamma
        self._epslion = epslion

    def train_step(self, env, epochs=100001):
        
        # 宣告 q-table
        q_table = np.zeros([env.observation_space.n, env.action_space.n])

        for i in tqdm(range(1, epochs)):
            state, _ = env.reset()

            epochs, reward, = 0, 0
            done = False

            while not done:
                
                if random.uniform(0, 1) < self._epslion:
                    action = env.action_space.sample() # 在一定的機率下，隨機選擇其他路線
                else:
                    action = np.argmax(q_table[state]) # 選擇價值最高的路線

                next_state, reward, done, _, _ = env.step(action) 

                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])
                
                new_value = (1 - self._alpha) * old_value + self._alpha * (reward + self._gamma * next_max)
                q_table[state, action] = new_value
                
                state = next_state
                epochs += 1
                
        return q_table