from tqdm import tqdm
from keras import Model, Sequential
from keras.layers import Dense, Embedding, Reshape
from keras.optimizers import Adam

import random
import numpy as np

class DeepQ_learning:
    def __init__(self, enviroment, optimizer, _gamma, _epsilon, _epsilon_min, _maxlen):
        
        # Initialize atributes
        self._state_size = enviroment.observation_space.n
        self._action_size = enviroment.action_space.n
        self._optimizer = optimizer
        
        self.expirience_replay = deque(maxlen=_maxlen)
        
        # Initialize discount and exploration rate
        self.gamma = _gamma
        self.epsilon = _epsilon
        self.epsilon_min = _epsilon_min
        self.epsilon_decay = 0.005
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1)) # Embedding層將每個不同的狀態表示成一個10維且唯一的狀態。
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        
        if random.uniform(0, 1) <= self.epsilon:
            print(f"The epsilon is {self.epsilon}, Random action")
            return env.action_space.sample()
        
        print(f"The epsilon is {self.epsilon}, Network action")
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def update_epsilon(self, episode):
        
        if self.epsilon > self.epsilon_min:
            self.epsilon= self.epsilon_min + (1-self.epsilon_min) * np.exp(-self.epsilon_decay * episode) 
    
    def retrain(self, batch_size):
        
        # 從過去的歷程中，隨機提取batch_size大小出來做訓練
        minibatch = random.sample(self.expirience_replay, batch_size)
        
        for state, action, reward, next_state, terminated in minibatch:
            
            target = self.q_network.predict(state)
            
            # print(target)
            
            if terminated:
                # print(reward)
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
                
            # print(target)
            
            self.q_network.fit(state, target, epochs=1, verbose=0)

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