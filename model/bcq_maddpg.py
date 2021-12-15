import torch
import numpy as np

from utils.misc import soft_update

from model.DDPGAgent import DDPGBCQAgent
from model.utils.model import *


class MADDPG(object):

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

        # Reshape critic input shape for shared observation
        params.critic.obs_dim = (self.obs_dim + self.action_dim) * self.num_agents

        self.agents = [DDPGBCQAgent(params) for _ in range(self.num_agents)]
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

            ''' Update VAE '''
            agent.vae_optimizer.zero_grad()

            recon, mean, std = agent.vae(obses[:, agent_i], actions[:, agent_i])

            if self.discrete_action:
                recon_loss = torch.nn.CrossEntropyLoss()(recon,  onehot_to_number(actions[:, agent_i]))
            else:
                recon_loss = F.mse_loss(recon, actions[:, agent_i])
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            vae_loss.backward()
            agent.vae_optimizer.step()

            ''' Update value '''

        for agent_i, agent in enumerate(self.agents):

            agent.critic_optimizer.zero_grad()

            with torch.no_grad():

                def get_perturbed_action(vae, policy, critic, state):
                    recon_action = vae.decode(state)
                    perturbed_action = policy(state, recon_action)
                    return perturbed_action

                # TODO State duplication
                if self.discrete_action:
                    target_actions = torch.Tensor([onehot_from_logits(get_perturbed_action(vae, policy, critic, next_obs)).detach().cpu().numpy()
                                                   for vae, policy, critic, next_obs in
                                               zip(self.vaes, self.target_policies, self.target_critics, torch.swapaxes(next_obses, 0, 1))]).to(self.device)

                else:
                    raise NotImplementedError('Overcooked only support discrete action space')
                    target_actions = torch.Tensor([policy(next_obs).detach().cpu().numpy() for policy, next_obs in
                                                   zip(self.target_policies, torch.swapaxes(next_obses, 0, 1))]).to(self.device)

                target_actions = torch.swapaxes(target_actions, 0, 1)
                target_critic_in = torch.cat((next_obses, target_actions), dim=2).view(self.batch_size, -1)
                target_next_q = rewards[:, agent_i] + (1 - dones[:, agent_i]) * self.gamma * agent.target_critic(target_critic_in)

            critic_in = torch.cat((obses, actions), dim=2).view(self.batch_size, -1)
            main_q = agent.critic(critic_in)

            critic_loss = self.mse_loss(main_q, target_next_q)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()

            ''' Update policy '''
            agent.policy_optimizer.zero_grad()

            sampled_actions = agent.vae.decode(obses[:, agent_i])
            policy_out = agent.policy(obses[:, agent_i], sampled_actions)

            if self.discrete_action:
                action = gumbel_softmax(policy_out, hard=True)
            else:
                action = policy_out

            joint_actions = torch.zeros((self.batch_size, self.num_agents, self.action_dim)).to(self.device)
            for i, policy, local_obs, act in zip(range(self.num_agents), self.policies, torch.swapaxes(obses, 0, 1), torch.swapaxes(actions, 0, 1)):
                if i == agent_i:
                    joint_actions[:, i] = action
                else:
                    joint_actions[:, i] = act

            critic_in = torch.cat((obses, joint_actions), dim=2).view(self.batch_size, -1)

            actor_loss = -agent.critic(critic_in).mean()
            actor_loss += (policy_out ** 2).mean() * 1e-3  # Action regularize
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
            agent.policy_optimizer.step()

        self.update_all_targets()

    def update_all_targets(self):
        for agent in self.agents:
            soft_update(agent.target_critic, agent.critic, self.tau)
            soft_update(agent.target_policy, agent.policy, self.tau)

    def save(self, step):
        # os.mk
        #
        # for i, agent in self.agents:
        #     name = '{0}_{1}_{step}.pth'.format(self.name, i, step)
        #     torch.save(agent, )
        #
        #
        # raise NotImplementedError
        pass

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

    @property
    def vaes(self):
        return [agent.vae for agent in self.agents]

