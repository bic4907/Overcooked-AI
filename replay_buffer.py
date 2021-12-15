import numpy as np
import torch
import json, os

from utils.train import make_dir
import logging

logger = logging.getLogger(__name__)

class ReplayBuffer(object):

    def __init__(self, obs_shape, action_shape, reward_shape, dones_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, *reward_shape), dtype=np.float32)
        self.dones = np.empty((capacity, *dones_shape), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, dones):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], dones)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, nth=None):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)

        if nth:
            obses = torch.FloatTensor(self.obses[idxs][:, nth]).to(self.device)
            actions = torch.FloatTensor(self.actions[idxs][:, nth]).to(self.device)
            rewards = torch.FloatTensor(self.rewards[idxs][:, nth]).to(self.device)
            next_obses = torch.FloatTensor(self.next_obses[idxs][:, nth]).to(self.device)
            dones = torch.FloatTensor(self.dones[idxs][:, nth]).to(self.device)
        else:
            obses = torch.FloatTensor(self.obses[idxs]).to(self.device)
            actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
            rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
            next_obses = torch.FloatTensor(self.next_obses[idxs]).to(self.device)
            dones = torch.FloatTensor(self.dones[idxs]).to(self.device)

        return obses, actions, rewards, next_obses, dones

    def save(self, root_dir, step) -> None:
        make_dir(root_dir, 'buffer') if root_dir else None
        length = self.capacity if self.full else self.idx
        path = os.path.join(root_dir, 'buffer')

        make_dir(path, str(step))
        path = os.path.join(path, str(step))

        np.savez_compressed(os.path.join(path, 'state.npz'), self.obses)
        np.savez_compressed(os.path.join(path, 'next_state.npz'), self.next_obses)
        np.savez_compressed(os.path.join(path, 'action.npz'), self.actions)
        np.savez_compressed(os.path.join(path, 'reward.npz'), self.rewards)
        np.savez_compressed(os.path.join(path, 'done.npz'), self.dones)

        info = dict()
        info['idx'] = self.idx
        info['capacity'] = self.capacity
        info['step'] = step
        info['size'] = length

        with open(os.path.join(path, 'info.txt'), 'w') as f:
            output = json.dumps(info, indent=4, sort_keys=True)
            f.write(output)

    def load(self, root_dir) -> None:
        path = os.path.join(root_dir, 'buffer')

        self.obses = np.load(os.path.join(path, 'state.npz'))['arr_0']
        self.next_obses = np.load(os.path.join(path, 'next_state.npz'))['arr_0']
        self.actions = np.load(os.path.join(path, 'action.npz'))['arr_0']
        self.rewards = np.load(os.path.join(path, 'reward.npz'))['arr_0']
        self.dones = np.load(os.path.join(path, 'done.npz'))['arr_0']

        with open(os.path.join(path, 'info.txt'), 'r') as f:
            info = json.load(f)

        self.idx = int(info['idx'])
        self.capacity = int(info['capacity'])
        self.full = int(info['step']) >= self.capacity

    def append_data(self, dir_path):

        def loader(path):
            logger.info('Loading data - ' + path)
            data =  np.load(path)['arr_0']
            logger.info('Loaded data - ' + path)
            return data

        obses_data = loader(os.path.join(dir_path, 'state.npz'))
        self.obses = np.concatenate((self.obses, obses_data), axis=0)

        next_obses_data = loader(os.path.join(dir_path, 'next_state.npz'))
        self.next_obses = np.concatenate((self.next_obses, next_obses_data), axis=0)

        reward_data = loader(os.path.join(dir_path, 'reward.npz'))
        self.rewards = np.concatenate((self.rewards, reward_data), axis=0)

        action_data = loader(os.path.join(dir_path, 'action.npz'))
        self.actions = np.concatenate((self.actions, action_data), axis=0)

        done_data = loader(os.path.join(dir_path, 'done.npz'))
        self.dones = np.concatenate((self.dones, done_data), axis=0)

        if self.idx == 0:
            self.idx = -1
        self.idx += len(obses_data)
