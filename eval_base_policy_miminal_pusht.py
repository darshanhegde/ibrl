
import torch
from lerobot.common.envs.utils import preprocess_observation

from env.pusht_wrapper import PushtWrapper
#from train_residual_rl import make_base_policy




def main():
    env = PushtWrapper(obs_type="environment_state_agent_pos", render_mode="human")
    #policy = make_base_policy("lerobot/diffusion_pusht_keypoints")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs, info = env.reset()
    for _ in range(300):
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        #print(obs)
        #observation = preprocess_observation(obs["raw_obs"])
        
        #observation = {key: observation[key].unsqueeze(0).to(device, non_blocking=True) for key in observation}

        with torch.inference_mode():
            action = torch.zeros(env.action_dim)
            #print(action)
            
            #action = policy.select_action(observation)

        obs, _, _, _, info = env.step(action)
        env.render()



if __name__ == "__main__":
    main()