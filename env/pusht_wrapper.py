import gymnasium as gym
import gym_pusht

import torch


class PushtWrapper:

    def __init__(self, obs_type, device='cuda', env_reward_scale=1.0, end_on_success=True): 
        self.obs_type = obs_type
        self.device = device
        self.env = gym.make("gym_pusht/PushT-v0", obs_type=obs_type)
        self.env_reward_scale = env_reward_scale
        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.end_on_success = end_on_success
        self.terminal = True
        self.max_steps = 400

    @property
    def observation_shape(self):
        return self.env.observation_space.shape
    
    @property
    def prop_shape(self):
        return self.env.action_space.shape
    
    @property
    def action_dim(self):
        return self.env.action_space.shape[0]
    
    def reset(self): 
        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False
    
        state_obs, info = self.env.reset()

        rl_obs = {}
        state_obs = torch.from_numpy(state_obs).float().to(self.device)
        rl_obs["state"] = state_obs
        return rl_obs, rl_obs
    
    def step(self, actions: torch.Tensor) -> tuple[dict, float, bool, bool, dict]:
        """
        all inputs and outputs are tensors
        """
        num_action = actions.size(0)
        actions = actions.numpy()

        reward = 0
        success = False
        terminal = False
        rl_obs = {}
        high_res_images = {}
        for i in range(num_action):
            self.time_step += 1
            obs, step_reward, terminal, _, info = self.env.step(actions[i])
            # NOTE: extract images every step for potential obs stacking
            # this is not efficient
            curr_rl_obs, curr_high_res_images = {"state": torch.from_numpy(obs).float().to(self.device)}, {"state": torch.from_numpy(obs).float().to(self.device)}
            if i == num_action - 1:
                rl_obs.update(curr_rl_obs)
                high_res_images.update(curr_high_res_images)

            reward += step_reward
            self.episode_reward += step_reward

            if step_reward == 1:
                success = True
                if self.end_on_success:
                    terminal = True

            if terminal:
                break

        if self.time_step >= self.max_steps:
            terminal = True

        reward = reward * self.env_reward_scale
        self.terminal = terminal
        return rl_obs, reward, terminal, success, high_res_images