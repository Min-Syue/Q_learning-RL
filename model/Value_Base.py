from tqdm import tqdm
from keras import Model, Sequential
from keras.layers import Dense, Embedding, Reshape
from keras.optimizers import Adam
from collections import deque

import random
import numpy as np
import time

class DeepQ_learning:
    def __init__(self, enviroment, optimizer, gamma=0.9, epsilon=1, epsilon_min=0.3, epsilon_decay_number=0.005, maxlen=200):
        
        # Initialize atributes
        self._state_size = enviroment.observation_space.n
        self._action_size = enviroment.action_space.n
        self._optimizer = optimizer
        self._env = enviroment
        
        # 設定要存多少數據
        self.expirience_replay = deque(maxlen=maxlen)
        
        # 初始化參數
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay_number
        
        # 建立神經網路模型
        self._q_network = self.build_compile_model()
        self._target_network = self.build_compile_model()
        self.alighn_target_model()

    # 紀錄每次運作的數據
    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))
    
    def build_compile_model(self):
        
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1)) # Embedding層將每個不同的狀態表示成一個10維且唯一的狀態。
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=self._optimizer)

        model.summary()

        return model

    # 將q_network的參數，設定給target_network
    def alighn_target_model(self):
        self._target_network.set_weights(self._q_network.get_weights())
    
    # agent的行動，有一定機率會跳脫原本思路，隨機選取
    def act(self, state):
        
        if random.uniform(0, 1) <= self._epsilon:
            return self._env.action_space.sample()
        
        q_values = self._q_network.predict(state)
        return np.argmax(q_values[0])

    # 這個是網路上找到的一個招，說要遞減每次跳脫原本思路的機率
    def update_epsilon(self, episode):
        
        if self.epsilon > self._epsilon_min:
            self.epsilon= self._epsilon_min + (1-self._epsilon_min) * np.exp(-self._epsilon_decay * episode) 
    
    # 開始模擬環境且訓練模型
    def train_step(self, num_of_episodes, alighn_weights, limit_stop=200, batch_size=32, training_steps=15, epsilon_decay=False):

        time_start = time.time()

        history_per_epiReward = []

        for e in tqdm(range(0, num_of_episodes)):
            
            # 重設gym的環境
            state, _ = self._env.reset()
            # print(state)
            state = np.reshape(state, [1, 1])
            
            # 初始化參數
            reward = 0
            total_reward = 0
            terminated = False
            
            for timestep in range(limit_stop):

                # Run Action
                action = self.act(state)

                # Take action    
                next_state, reward, terminated, _, _ = self._env.step(action) 
                
                #if next_state == state and action != 5 and action != 6:
                #   reward = -5
                
                total_reward += reward
                 
                next_state = np.reshape(next_state, [1, 1])
                self.store(state, action, reward, next_state, terminated)
                
                state = next_state
           
                if len(self.expirience_replay) > batch_size and timestep % training_steps == 0:
                    self.retrain(batch_size)
                
                if timestep % alighn_weights == 0:
                    self.alighn_target_model()
                
            history_per_epiReward.append(total_reward)
            
            self.alighn_target_model()
            
            if epsilon_decay:
                self.update_epsilon(e)
            
        time_end = time.time()

        time_c= time_end - time_start 
        
        return history_per_epiReward, time_c

    # 訓練模型的實際步驟
    def retrain(self, batch_size):
        
        # 從過去的歷程中，隨機提取batch_size大小出來做訓練
        minibatch = random.sample(self.expirience_replay, batch_size)
        
        for state, action, reward, next_state, terminated in minibatch:
            
            target = self._q_network.predict(state, verbose=0)
            
            # print(target)
            
            if terminated:
                # print(reward)
                target[0][action] = reward
            else:
                t = self._target_network.predict(next_state, verbose=0)
                target[0][action] = reward + self._gamma * np.amax(t)
                
            # print(target)
            
            self._q_network.fit(state, target, epochs=1, verbose=0)

class Sarsa:

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

            epochs, reward, done = 0, 0, False
            action = env.action_space.sample()

            while not done:
                
                next_state, reward, done, _, _ = env.step(action)

                if random.uniform(0, 1) < self._epslion:
                    next_action = env.action_space.sample() # 在一定的機率下，隨機選擇其他路線
                else:
                    next_action = np.argmax(q_table[next_state]) # 選擇價值最高的路線

                old_value = q_table[state, action]
                next_value = q_table[next_state, next_action]
                
                new_value = (1 - self._alpha) * old_value + self._alpha * (reward + self._gamma * next_value)
                q_table[state, action] = new_value
                
                state = next_state
                action = next_action
                epochs += 1
                
        return q_table


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