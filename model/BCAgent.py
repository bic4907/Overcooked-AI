from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn

from model.network import BCNetwork
from model.utils.model import *
from model.utils.noise import OUNoise


class BCAgent(nn.Module):
    """
    General class for BC agents (policy, exploration noise)
    """

    def __init__(self, params):
        super(BCAgent, self).__init__()

        self.lr = params.lr
        self.gamma = params.gamma

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.device = params.device
        self.discrete_action = params.discrete_action_space
        self.hidden_dim = params.hidden_dim

        constrain_out = not self.discrete_action

        self.policy = BCNetwork(self.obs_dim, self.action_dim, hidden_dim=self.hidden_dim)
        # self.target_policy = BCNetwork(self.obs_dim, self.action_dim, hidden_dim=self.hidden_dim)

        # hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr, eps=1e-08)

        # self.exploration = OUNoise(self.action_dim)

        self.num_heads = 100

    def act(self, obs, explore=False):

        if obs.dim() == 1:
            obs = obs.unsqueeze(dim=0)

        action = self.policy(obs)
        action = onehot_from_logits(action, eps=0.0)
        action = onehot_to_number(action)

        return action.detach().cpu().numpy()

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                # 'critic': self.critic.state_dict(),
                # 'target_policy': self.target_policy.state_dict(),
                # 'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                # 'critic_optimizer': self.critic_optimizer.state_dict()
                }