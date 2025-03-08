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
            snapshot_download(pretrained_policy_name, revision="24a053c0dc7659d300601cb3556a7d5618a9c757") #old data for hydra compatibility
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

class PushtImageWrapper:

    def __init__(self, obs_type, render_mode="rgb_array", device='cuda', env_reward_scale=1.0, end_on_success=True,
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
        self.max_steps = 300

        self.base_policy = make_base_policy("lerobot/diffusion_pusht")
        self.next_action = None
        self.action_scale = env_action_scale

    @property
    def observation_shape(self):
        obs_space = self.env.observation_space.spaces
        image_shape = obs_space["pixels"].shape  # This is (H, W, C)
        
        # Convert to (C, H, W)
        corrected_shape = (image_shape[2], image_shape[0], image_shape[1])  # (C, H, W)
        
        print("Using observation shape:", corrected_shape)
        return corrected_shape

        
    
    @property
    def prop_shape(self):
        return self.env.observation_space.spaces["agent_pos"].shape #assuming that prop (agent_pos) is appended to image data in QAgent
    
    @property
    def action_dim(self):
        return self.env.action_space.shape[0]
    
    def run_base_policy(self, raw_obs: dict[str, np.array]):

        observation = preprocess_observation(raw_obs)
        observation = {key: (obs if obs.ndim >= 4 else obs.unsqueeze(0)).to(self.device, non_blocking=True)
                        for key, obs in observation.items()}

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
        concat_obs = np.concatenate([obs["agent_pos"], base_action])

        rl_obs = {"image": torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float().to(self.device)}
        rl_obs["state"] = torch.from_numpy(concat_obs).float().to(self.device) # (i think) diffusion expects end effector pos in state
        rl_obs["prop"] = torch.from_numpy(obs["agent_pos"]).float().to(self.device) # Qagent expects end effector pos is prop
        return rl_obs, info
    
    def step(self, actions: torch.Tensor) -> tuple[dict, float, bool, bool, dict]:
        """
        all inputs and outputs are tensors
        """
        actions = actions.to("cpu").numpy()

        # scale actions coming from the policy
        actions = actions * self.action_scale

        # print("Actions: ", actions)

        reward = 0
        success = False
        terminal = False
        info = {}
        rl_obs = {}
        self.time_step += 1
        final_action = self.next_action + actions

        obs, step_reward, terminal, _, info = self.env.step(final_action)

        base_action = self.run_base_policy(obs)
        self.next_action = base_action

        concat_obs = np.concatenate([obs["agent_pos"], base_action])
        curr_rl_obs = {"image": torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float().to(self.device)}
        curr_rl_obs["state"] = torch.from_numpy(concat_obs).float().to(self.device)
        rl_obs["prop"] = torch.from_numpy(obs["agent_pos"]).float().to(self.device) # Qagent expects end effector pos is prop
        rl_obs.update(curr_rl_obs)

        if step_reward >= 0.95:
            step_reward = 1
            success = True
            if self.end_on_success:
                terminal = True
        else: 
            step_reward = 0

        reward += step_reward
        self.episode_reward += step_reward

        if self.time_step >= self.max_steps:
            terminal = True

        reward = reward * self.env_reward_scale
        self.terminal = terminal
        return rl_obs, reward, terminal, success, info
        # num_action = actions.size(0)
        # actions = actions.to("cpu").numpy()
        # actions = actions * self.action_scale

        # reward = 0
        # success = False
        # terminal = False
        # info = {}
        # rl_obs = {}
        # for i in range(num_action):
        #     self.time_step += 1
        #     final_action = self.next_action + actions[i]
        #     obs, step_reward, terminal, _, info = self.env.step(final_action)

        #     base_action = self.run_base_policy(obs)
        #     self.next_action = base_action

        #     concat_obs = np.concatenate([obs["agent_pos"], base_action])
        #     curr_rl_obs = {"image": torch.from_numpy(obs["pixels"]).float().to(self.device)}
        #     curr_rl_obs["state"] = torch.from_numpy(concat_obs).float().to(self.device)
        #     if i == num_action - 1:
        #         rl_obs.update(curr_rl_obs)
                

        #     reward += step_reward
        #     self.episode_reward += step_reward

        #     if step_reward == 1:
        #         success = True
        #         if self.end_on_success:
        #             terminal = True

        #     if terminal:
        #         rl_obs.update(curr_rl_obs)
        #         break

        # if self.time_step >= self.max_steps:
        #     terminal = True

        # reward = reward * self.env_reward_scale
        # self.terminal = terminal
        # return rl_obs, reward, terminal, success, info
    
    def render(self):
        self.env.render()  # Should return an RGB array
    

def main():
    env = PushtImageWrapper("pixels_agent_pos", render_mode="rgb_array")
    obs, _ = env.reset()
    print(obs['state'].shape)
    env.render()
    for _ in range(10):
        obs, reward, terminal, success, _ = env.step(torch.zeros(2))
        print(obs['state'].shape)
        env.render()
        if terminal:
            break

if __name__ == "__main__":
    main()