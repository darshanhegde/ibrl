
import torch
import numpy as np
from lerobot.common.envs.utils import preprocess_observation
from dataclasses import dataclass, field
import copy

from env.pusht_wrapper import PushtWrapper
from rl.q_agent import QAgent, QAgentConfig
from rl.critic import MultiFcQ, MultiFcQConfig
from rl.actor import FcActor, FcActorConfig


def main():
    env = PushtWrapper(obs_type="environment_state_agent_pos", env_action_scale=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mcfg = MultiFcQConfig(hidden_dim=1024, num_q=5, layer_norm=1)
    mfcfg = FcActorConfig(hidden_dim=1024, dropout=0.5)

    agent_cfg = QAgentConfig(enc_type="vit", state_critic=mcfg, state_actor=mfcfg)
    print(agent_cfg)

    """
    state_critic:
    num_q: 5
    layer_norm: 1
    hidden_dim: 1024
    state_actor:
    hidden_dim: 1024
    dropout: 0
    """

    agent = QAgent(
        1,
        env.observation_shape,
        env.prop_shape,
        env.action_dim,
        "robot0_eye_in_hand",
        agent_cfg,
    )

    path = "exps/rl/push_t_keypoints_ibrl_scale_4_max_dev_1/latest.pt"
    print(f"loading loading pretrained agent from {path}")
    critic_states = copy.deepcopy(agent.critic.state_dict())
    agent.load_state_dict(torch.load(path))

    agent.critic.load_state_dict(critic_states)
    agent.critic_target.load_state_dict(critic_states)

    agent.actor.training = False
    target = torch.tensor([-1., -1.])

    num_success = 0
    num_episodes = 200
    for j in range(num_episodes):
        obs, info = env.reset()
        for i in range(300):
            sample_action = np.array([0, 0])
            sample_action = torch.from_numpy(sample_action).float().to(device)

            # agent.training = False
            # agent.actor.training = False
            # action = agent.act(obs, eval_mode=True)
            # if torch.equal(action, target):
            #     print(f"Action: {action}")
            # #print(f" - Action: {action} | Obs: {obs}")

            obs, reward, terminal, success, info = env.step(sample_action)
            if reward > 0:
                print(f"{j}: Success")
            env.render()

            if terminal:
                if success:
                    num_success += success
                else: 
                    print(f"{j}: Failed")
                break
            
    print("Percent success: ", num_success / num_episodes)


if __name__ == "__main__":
    main()