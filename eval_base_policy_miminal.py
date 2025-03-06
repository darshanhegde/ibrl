
import torch
import numpy as np
from lerobot.common.envs.utils import preprocess_observation

from env.pusht_wrapper import PushtWrapper


def main():
    env = PushtWrapper(obs_type="environment_state_agent_pos", render_mode="human", env_action_scale=16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_success = 0
    num_episodes = 100
    for _ in range(num_episodes):
        obs, info = env.reset()
        for _ in range(300):
            sample_action = np.random.random((2,)) * 0.0
            sample_action = torch.from_numpy(sample_action).float().to(device)
            obs, reward, terminal, success, info = env.step(sample_action)
            if reward > 0:
                print("Success")
            env.render()

            if terminal:
                if success:
                    num_success += success
                else: 
                    print("Failed")
                break
            
    print("Percent success: ", num_success / num_episodes)


if __name__ == "__main__":
    main()