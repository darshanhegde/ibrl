
import torch
from lerobot.common.envs.utils import preprocess_observation

from env.aloha_wrapper import AlohaInsertionWrapper
#from train_residual_rl import make_base_policy




def main():
    env = AlohaInsertionWrapper(obs_type="environment_state_agent_pos", render_mode="human")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs, info = env.reset()
    for _ in range(300):
        with torch.inference_mode():
            action = torch.zeros(env.action_dim)

        obs, _, _, _, info = env.step(action)
        env.render()



if __name__ == "__main__":
    main()