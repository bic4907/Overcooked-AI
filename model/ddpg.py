import torch
import numpy as np

from utils.misc import soft_update

from model.DDPGAgent import DDPGAgent
from model.utils.model import *


class DDPG(object):

    def __init__(self, name, params):

        self.name = name
        self.lr = params.lr
        self.gamma = params.gamma
        self.tau = params.tau

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.batch_size = params.batch_size
        self.device = params.device
        self.discrete_action = params.discrete_action_space

        self.agent_index = params.agent_index
        self.num_agents = len(self.agent_index)

        self.mse_loss = torch.nn.MSELoss()

        params.critic.obs_dim = (self.obs_dim + self.action_dim)

        self.agents = [DDPGAgent(params) for _ in range(self.num_agents)]
        [agent.to(self.device) for agent in self.agents]

    def scale_noise(self, scale):
        for agent in self.agents:
            agent.scale_noise(scale)

    def reset_noise(self):
        for agent in self.agents:
            agent.reset_noise()

    def act(self, observations, sample=False):
        observations = torch.Tensor(observations).to(self.device)

        actions = []
        for agent, obs in zip(self.agents, observations):
            agent.eval()
            actions.append(agent.act(obs, explore=sample).squeeze())
            agent.train()
        return np.array(actions)

    def update(self, replay_buffer, logger, step):

        sample = replay_buffer.sample(self.batch_size, nth=self.agent_index)
        obses, actions, rewards, next_obses, dones = sample

        if self.discrete_action:  actions = number_to_onehot(actions)

        for agent_i, agent in enumerate(self.agents):

            ''' Update value '''
            agent.critic_optimizer.zero_grad()

            with torch.no_grad():
                if self.discrete_action:
                    target_action = onehot_from_logits(agent.policy(next_obses[:, agent_i]))
                else:
                    target_action= agent.policy(next_obses[:, agent_i])
                target_critic_in = torch.cat((next_obses[:, agent_i], target_action), dim=1)
                target_next_q = rewards[:, agent_i] + (1 - dones[:, agent_i]) * self.gamma * agent.target_critic(target_critic_in)

            critic_in = torch.cat((obses[:, agent_i], actions[:, agent_i]), dim=1)
            main_q = agent.critic(critic_in)

            critic_loss = self.mse_loss(main_q, target_next_q)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()

            ''' Update policy '''
            agent.policy_optimizer.zero_grad()

            if self.discrete_action:
                action = gumbel_softmax(agent.policy(obses[:, agent_i]), hard=True)
            else:
                action = agent.policy(obses[:, agent_i])

            critic_in = torch.cat((obses[:, agent_i], action), dim=1)

            actor_loss = -agent.critic(critic_in).mean()
            actor_loss += (action ** 2).mean() * 1e-3  # Action regularize
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[agent_i].policy.parameters(), 0.5)
            agent.policy_optimizer.step()

        self.update_all_targets()

    def update_all_targets(self):
        for agent in self.agents:
            soft_update(agent.target_critic, agent.critic, self.tau)
            soft_update(agent.target_policy, agent.policy, self.tau)

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

    @property
    def policies(self):
        return [agent.policy for agent in self.agents]

    @property
    def target_policies(self):
        return [agent.target_policy for agent in self.agents]

    @property
    def critics(self):
        return [agent.critic for agent in self.agents]

    @property
    def target_critics(self):
        return [agent.target_critic for agent in self.agents]
