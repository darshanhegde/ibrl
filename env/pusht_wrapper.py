import gymnasium as gym
import gym_pusht

import logging
import numpy as np
import torch
from pathlib import Path

from lerobot.common.envs.utils import preprocess_observation

from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
from lerobot.common.policies.factory import make_policy


def make_base_policy(pretrained_policy_name):
    try:
        pretrained_policy_path = Path(
            snapshot_download(pretrained_policy_name, revision=None)
        )
    except (HFValidationError, RepositoryNotFoundError) as e:
        if isinstance(e, HFValidationError):
            error_message = (
                "The provided pretrained_policy_name is not a valid Hugging Face Hub repo ID."
            )
        else:
            error_message = (
                "The provided pretrained_policy_name was not found on the Hugging Face Hub."
            )
        logging.warning(f"{error_message} Treating it as a local directory.")
        if not pretrained_policy_path.is_dir() or not pretrained_policy_path.exists():
            raise ValueError(
                "The provided pretrained_policy_name_or_path is not a valid/existing Hugging Face Hub "
                "repo ID, nor is it an existing local directory."
            )
        
    hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"))
    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))
    policy.eval()
    return policy

class PushtWrapper:

    def __init__(self, obs_type, render_mode="rgb_array", device='cuda', 
                 env_reward_scale=1.0, end_on_success=True, 
                 env_action_scale=16): 
        self.obs_type = obs_type
        self.device = device
        self.env = gym.make("gym_pusht/PushT-v0", obs_type=obs_type, render_mode=render_mode)
        self.env_reward_scale = env_reward_scale
        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.end_on_success = end_on_success
        self.terminal = True
        self.max_steps = 400

        self.base_policy = make_base_policy("lerobot/diffusion_pusht_keypoints")
        self.next_action = None
        self.action_scale = env_action_scale

    @property
    def observation_shape(self):
        # loop thrpough observation_space keys and add up box dimnensions
        obs_space = self.env.observation_space.spaces
        observation_shape = (obs_space["environment_state"].shape[0] + 2 * obs_space["agent_pos"].shape[0],) 
        print("Using observation shape: ", observation_shape)
        return observation_shape
        
    
    @property
    def prop_shape(self):
        return self.env.action_space.shape
    
    @property
    def action_dim(self):
        return self.env.action_space.shape[0]
    
    def run_base_policy(self, raw_obs: dict[str, np.array]):
        observation = preprocess_observation(raw_obs)
        observation = {key: observation[key].unsqueeze(0).to(self.device, non_blocking=True) for key in observation}

        with torch.inference_mode():
            action = self.base_policy.select_action(observation)
            return action.squeeze(0).to("cpu").numpy()
    
    def reset(self): 
        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False
    
        self.base_policy.reset()
        obs, info = self.env.reset()
        base_action = self.run_base_policy(obs)
        self.next_action = base_action
        # concatenate all observations
        concat_obs = np.concatenate([obs["environment_state"], obs["agent_pos"], base_action])

        rl_obs = {}
        rl_obs["state"] = torch.from_numpy(concat_obs).float().to(self.device)
        return rl_obs, info
    
    def step(self, actions: torch.Tensor) -> tuple[dict, float, bool, bool, dict]:
        """
        all inputs and outputs are tensors
        """
        num_action = actions.size(0)
        actions = actions.to("cpu").numpy()

        # scale actions coming from the policy
        actions = (actions + 1) * self.action_scale

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

            concat_obs = np.concatenate([obs["environment_state"], obs["agent_pos"], base_action])
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
    env = PushtWrapper("environment_state_agent_pos", render_mode="human")
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