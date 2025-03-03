# NOTE: this is for aloha keypoints (not image, since no pre-trained model exists for image atm)

import gymnasium as gym
import gym_aloha
import logging
import numpy as np
import torch
from pathlib import Path

from lerobot.common.envs.utils import preprocess_observation
from huggingface_hub import snapshot_download
#from huggingface_hub.utils._errors import RepositoryNotFoundError
#from huggingface_hub.utils._validators import HFValidationError
from lerobot.common.utils.utils import get_safe_torch_device #init_hydra_config, init_logging, set_global_seed
from lerobot.common.policies.factory import make_policy

from transformers import AutoModel

def make_base_policy(pretrained_policy_name, device):
    #try:
    pretrained_policy_path = Path(snapshot_download(pretrained_policy_name, revision=None))
    print(pretrained_policy_path)
    # except (HFValidationError, RepositoryNotFoundError) as e:
    #     if isinstance(e, HFValidationError):
    #         error_message = "Invalid Hugging Face Hub repo ID."
    #     else:
    #         error_message = "Repo not found on Hugging Face Hub."
    #     logging.warning(f"{error_message} Treating it as a local directory.")
    #     if not pretrained_policy_path.is_dir() or not pretrained_policy_path.exists():
    #         raise ValueError("Invalid Hugging Face repo or nonexistent local directory.")

    # NOTE: current implementation does NOT use diffusion/have config.yaml so no hydra needed
    # hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"))
    # policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))

    from omegaconf import OmegaConf

    # Create an empty configuration
    hydra_cfg = OmegaConf.create()

    # Set the desired policy name
    hydra_cfg.policy = OmegaConf.create()
    hydra_cfg.policy.name = "act"
    hydra_cfg.device = device

    device = get_safe_torch_device(device)
    policy = make_policy(
            hydra_cfg=hydra_cfg,
            pretrained_policy_name_or_path=pretrained_policy_name,
        )
    policy.eval()
    return policy


class AlohaInsertionWrapper:
    def __init__(self, obs_type, render_mode="rgb_array", device='cuda', env_reward_scale=1.0, end_on_success=True): 
        self.obs_type = obs_type
        self.device = device
        self.env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels_agent_pos", render_mode=render_mode)  # Change env
        self.env_reward_scale = env_reward_scale
        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.end_on_success = end_on_success
        self.terminal = True
        self.max_steps = 400

        self.base_policy = make_base_policy("lerobot/act_aloha_sim_insertion_human", device)  # Change policy
        self.next_action = None

    @property
    def observation_shape(self):
        obs_space = self.env.observation_space.spaces
        observation_shape = (obs_space["pixels"]["top"].shape[0] + 2 * obs_space["agent_pos"].shape[0],) 
        print("Using observation shape: ", observation_shape)
        return observation_shape
        
    @property
    def prop_shape(self):
        return self.env.action_space.shape
    
    @property
    def action_dim(self):
        return self.env.action_space.shape[0]
    
    def run_base_policy(self, raw_obs: dict[str, np.array]):
        #print(f"Raw: {raw_obs}")
        observation = preprocess_observation(raw_obs)
        print(f"KEYS after preprocess: {observation.keys()}")
        observation = {key: observation[key].unsqueeze(0).to(self.device, non_blocking=True) for key in observation}

        with torch.inference_mode():
            action = self.base_policy.select_action(observation)
            return action.squeeze(0).to("cpu").numpy()
    
    def reset(self): 
        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False
    
        # NOTE: removed since ViT has no attribute 'reset'
        #self.base_policy.reset()
        obs, info = self.env.reset()
        observation = preprocess_observation(obs)
        print(observation.keys())

        print(obs.keys())
        base_action = self.run_base_policy(obs)
        self.next_action = base_action
        print(obs["pixels"]["top"].shape)
        print(obs["agent_pos"].shape)
        print(base_action.shape)
        print(obs.keys())
        print(observation["observation.state"])
        concat_obs = np.concatenate([observation["observation.state"], obs["agent_pos"], base_action])
        print(concat_obs.shape)

        rl_obs = {}
        rl_obs["state"] = torch.from_numpy(concat_obs).float().to(self.device)
        return rl_obs, info
    
    def step(self, actions: torch.Tensor) -> tuple[dict, float, bool, bool, dict]:
        """
        all inputs and outputs are tensors
        """
        num_action = actions.size(0)
        actions = actions.to("cpu").numpy()

        reward = 0
        success = False
        terminal = False
        info = {}
        rl_obs = {}
        for i in range(num_action):
            self.time_step += 1
            final_action = self.next_action + actions[i]
            obs, step_reward, terminal, _, info = self.env.step(final_action)

            base_action = self.run_base_policy(obs)
            self.next_action = base_action

            print(obs["pixels"]["top"].shape)
            print(obs["agent_pos"].shape)
            observation = preprocess_observation(obs)


            concat_obs = np.concatenate([observation["observation.state"], obs["agent_pos"], base_action])
            curr_rl_obs = {}
            curr_rl_obs["state"] = torch.from_numpy(concat_obs).float().to(self.device)
            if i == num_action - 1:
                rl_obs.update(curr_rl_obs)
                
            reward += step_reward
            self.episode_reward += step_reward

            if step_reward == 1:
                success = True
                if self.end_on_success:
                    terminal = True

            if terminal:
                rl_obs.update(curr_rl_obs)
                break

        if self.time_step >= self.max_steps:
            terminal = True

        reward = reward * self.env_reward_scale
        self.terminal = terminal
        return rl_obs, reward, terminal, success, info
    
    def render(self):
        self.env.render()   
    

def main():
    env = AlohaInsertionWrapper("pixels_agent_pos", render_mode="human")
    obs, _ = env.reset()
    print(obs['state'].shape)
    env.render()
    for _ in range(10):
        obs, reward, terminal, success, _ = env.step(torch.zeros((1, 2)))
        print(obs['state'].shape)
        env.render()
        if terminal:
            break

if __name__ == "__main__":
    main()
