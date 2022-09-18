from gym.utils.save_video import save_video

import numpy as np

def agent_walk_path_vedio(env, q_table, file_name, limit_epoch=200):
    
    # 宣告參數
    done = False
    epochs = 0
    reward_t = 0

    env.reset()

    while not done and epochs < limit_epoch:
        
        # 現在環境的代號
        state = env.s
        
        # 現在最大q值的action
        action = np.argmax(q_table[state])

        state, reward, done, _, _ = env.step(action)

        epochs += 1
        reward_t += reward
        
    save_video(
            env.render(),
            file_name,
            fps=env.metadata["render_fps"],
            step_starting_index=0,
            episode_index=0
        )

    return epochs, reward_t