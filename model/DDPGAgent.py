from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn

from model.network import MLPNetwork, VAENetwork, BCQActorNetwork
from model.utils.model import *
from model.utils.noise import OUNoise


class DDPGAgent(nn.Module):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """

    def __init__(self, params):
        super(DDPGAgent, self).__init__()
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """

        self.lr = params.lr
        self.gamma = params.gamma

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.device = params.device
        self.discrete_action = params.discrete_action_space
        self.hidden_dim = params.hidden_dim

        constrain_out = not self.discrete_action

        self.policy = MLPNetwork(self.obs_dim, self.action_dim,
                                 hidden_dim=self.hidden_dim,
                                 constrain_out=constrain_out)
        self.critic = MLPNetwork(params.critic.obs_dim, 1,
                                 hidden_dim=self.hidden_dim,
                                 constrain_out=False)
        self.target_policy = MLPNetwork(self.obs_dim, self.action_dim,
                                        hidden_dim=self.hidden_dim,
                                        constrain_out=constrain_out)
        self.target_critic = MLPNetwork(params.critic.obs_dim, 1,
                                        hidden_dim=self.hidden_dim,
                                        constrain_out=False)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr * 0.1)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        self.exploration = OUNoise(self.action_dim)

        self.num_heads = 100

    def act(self, obs, explore=False):

        if obs.dim() == 1:
            obs = obs.unsqueeze(dim=0)

        action = self.policy(obs)

        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)

            action = onehot_to_number(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()), requires_grad=False).to(action.device)
            action = action.clamp(-1, 1)

        return action.detach().cpu().numpy()

    def reset_noise(self):
        self.exploration.reset()

    def scale_noise(self, scale):
        self.exploration.scale = scale

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])


class DDPGREMAgent(nn.Module):

    def __init__(self, params):
        super(DDPGREMAgent, self).__init__()
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """

        self.lr = params.lr
        self.gamma = params.gamma

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.device = params.device
        self.discrete_action = params.discrete_action_space
        self.hidden_dim = params.hidden_dim

        constrain_out = not self.discrete_action

        self.num_heads = 100

        self.policy = MLPNetwork(self.obs_dim, self.action_dim,
                                 hidden_dim=self.hidden_dim,
                                 constrain_out=constrain_out)
        self.critic = MLPNetwork(params.critic.obs_dim, 1 * self.num_heads,
                                 hidden_dim=self.hidden_dim,
                                 constrain_out=False)
        self.target_policy = MLPNetwork(self.obs_dim, self.action_dim,
                                        hidden_dim=self.hidden_dim,
                                        constrain_out=constrain_out)
        self.target_critic = MLPNetwork(params.critic.obs_dim, 1 * self.num_heads,
                                        hidden_dim=self.hidden_dim,
                                        constrain_out=False)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr * 0.1)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        self.exploration = OUNoise(self.action_dim)

    def act(self, obs, explore=False):

        if obs.dim() == 1:
            obs = obs.unsqueeze(dim=0)

        action = self.policy(obs)

        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)

            action = onehot_to_number(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()), requires_grad=False).to(action.device)
            action = action.clamp(-1, 1)

        return action.detach().cpu().numpy()

    def reset_noise(self):
        self.exploration.reset()

    def scale_noise(self, scale):
        self.exploration.scale = scale


class DDPGBCQAgent(nn.Module):

    def __init__(self, params):
        super(DDPGBCQAgent, self).__init__()
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """

        self.lr = params.lr
        self.gamma = params.gamma

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.device = params.device
        self.discrete_action = params.discrete_action_space
        self.hidden_dim = params.hidden_dim

        constrain_out = not self.discrete_action

        self.num_heads = 100

        self.policy = BCQActorNetwork(self.obs_dim + self.action_dim, self.action_dim,
                                 hidden_dim=self.hidden_dim,
                                 constrain_out=constrain_out)
        self.critic = MLPNetwork(params.critic.obs_dim, 1 * self.num_heads,
                                 hidden_dim=self.hidden_dim,
                                 constrain_out=False)
        self.target_policy = BCQActorNetwork(self.obs_dim + self.action_dim, self.action_dim,
                                        hidden_dim=self.hidden_dim,
                                        constrain_out=constrain_out)
        self.target_critic = MLPNetwork(params.critic.obs_dim, 1 * self.num_heads,
                                        hidden_dim=self.hidden_dim,
                                        constrain_out=False)

        self.target_critic = MLPNetwork(params.critic.obs_dim, 1 * self.num_heads,
                                        hidden_dim=self.hidden_dim,
                                        constrain_out=False)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr * 0.1)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        self.vae = VAENetwork(self.obs_dim, self.action_dim, self.action_dim * 2)
        self.vae_optimizer = Adam(self.vae.parameters(), lr=self.lr)

        self.exploration = OUNoise(self.action_dim)

    def act(self, obs, explore=False):

        if obs.dim() == 1:
            obs = obs.unsqueeze(dim=0)

        action = self.policy(obs, self.vae.decode(obs))

        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)

            action = onehot_to_number(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()), requires_grad=False).to(action.device)
            action = action.clamp(-1, 1)

        return action.detach().cpu().numpy()

    def reset_noise(self):
        self.exploration.reset()

    def scale_noise(self, scale):
        self.exploration.scale = scale






