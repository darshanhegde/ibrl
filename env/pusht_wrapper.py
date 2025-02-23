import gymnasium as gym
import gym_pusht

import numpy as np
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
        # loop thrpough observation_space keys and add up box dimnensions
        obs_space = self.env.observation_space.spaces
        return (obs_space["environment_state"].shape[0] + obs_space["agent_pos"].shape[0],)
        
    
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
        # concatenate all observations
        concat_obs = np.concatenate([state_obs["environment_state"], state_obs["agent_pos"]])

        rl_obs = {}
        rl_obs["state"] = torch.from_numpy(concat_obs).float().to(self.device)
        return rl_obs, state_obs
    
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
        curr_state_obs = {}
        for i in range(num_action):
            self.time_step += 1
            obs, step_reward, terminal, _, info = self.env.step(actions[i])
            concat_obs = np.concatenate([obs["environment_state"], obs["agent_pos"]])
            curr_rl_obs, curr_state_obs = {"state": torch.from_numpy(concat_obs).float().to(self.device)}, obs
            if i == num_action - 1:
                rl_obs.update(curr_rl_obs)
                curr_state_obs.update(curr_state_obs)

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
        return rl_obs, reward, terminal, success, curr_state_obs
    

def main():
    env = PushtWrapper("environment_state_agent_pos")
    obs, _ = env.reset()
    import pdb; pdb.set_trace()
    for _ in range(100):
        obs, reward, terminal, success, _ = env.step(torch.zeros((1, 2)))
        if terminal:
            break



if __name__ == "__main__":
    main()