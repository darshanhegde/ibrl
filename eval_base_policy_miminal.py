
import torch
import numpy as np
from lerobot.common.envs.utils import preprocess_observation

from env.pusht_wrapper import PushtWrapper


def main():
    env = PushtWrapper(obs_type="environment_state_agent_pos", render_mode="human")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs, info = env.reset()
    for _ in range(300):
        sample_action = -1.0 * np.ones((2,))
        sample_action = torch.from_numpy(sample_action).float().to(device)
        obs, reward, terminal, success, info = env.step(sample_action)
        if success:
            print("Success !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        env.render()



if __name__ == "__main__":
    main()