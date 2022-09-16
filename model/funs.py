from gym.utils.save_video import save_video

import numpy as np

def agent_walk_path_vedio(env, q_table, times=1):
    
    # 宣告參數
    output_epoch = []
    total_reward = []

    for i in range(times):

        done = False
        epochs = 0
        reward_t = 0

        env.reset()

        while not done:

            # 現在環境的代號
            state = env.s
        
            # 現在最大q值的action
            action = np.argmax(q_table[state])

            state, reward, done, _, _ = env.step(action)

            epochs += 1
            reward_t += reward
        
        output_epoch.append(epochs)
        total_reward.append(reward_t)

        save_video(
                env.render(),
                "videos",
                fps=env.metadata["render_fps"],
                step_starting_index=0,
                episode_index=i
            )

    return output_epoch, total_reward