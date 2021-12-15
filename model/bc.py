import torch
import numpy as np

from utils.misc import soft_update

from model.BCAgent import BCAgent
from model.utils.model import *


class BahaviorClone(object):

    def __init__(self, name, params):

        self.name = name
        self.lr = params.lr
        self.gamma = params.gamma
        self.tau = params.tau

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.batch_size = params.batch_size // 2
        self.device = params.device
        self.discrete_action = params.discrete_action_space

        self.agent_index = params.agent_index
        self.num_agents = len(self.agent_index)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        params.critic.obs_dim = (self.obs_dim + self.action_dim)

        self.agents = [BCAgent(params) for _ in range(self.num_agents)]
        [agent.to(self.device) for agent in self.agents]

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

        # split each joint into two single trajectories
        # need to check
        obses = (torch.cat([obses[:, 0], obses[:, 1]], dim=0))
        actions = (torch.cat([actions[:, 0], actions[:, 1]], dim=0))
        rewards = (torch.cat([rewards[:, 0], rewards[:, 1]], dim=0))
        next_obses = (torch.cat([next_obses[:, 0], next_obses[:, 1]], dim=0))
        dones = (torch.cat([dones[:, 0], dones[:, 1]], dim=0))

        if self.discrete_action:
            actions = number_to_onehot(actions)
            actions = torch.max(actions.long(), 1)[1]

        for agent_i, agent in enumerate(self.agents):
            agent.policy_optimizer.zero_grad()
            agent_actions = agent.policy(obses)
            loss = self.cross_entropy_loss(agent_actions, actions)
            loss.backward()
            agent.policy_optimizer.step()


    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

