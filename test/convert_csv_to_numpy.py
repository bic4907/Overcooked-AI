import pickle
import json
import numpy as np

from overcooked_ai_py.env import OverCookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from overcooked_ai_py.mdp.overcooked_mdp import Recipe

from tqdm import tqdm

def _convert_action_to_int(joint_action) -> list:
    action_set = []
    for _action in joint_action:
        if _action == (-1, 0): #
            action_set.append(3)
        elif _action == (0, 1): # down
            action_set.append(1)
        elif _action == (1, 0): # right
            action_set.append(2)
        elif _action == (0, -1): # Up
            action_set.append(0)
        elif _action == 'interact':
            action_set.append(5)
        else:
            action_set.append(4)

    return action_set


from replay_buffer import ReplayBuffer

# Load data
with open('../overcooked_ai_py/data/human_data/clean_train_trials.pickle', 'rb') as f:
    dfA = pickle.load(f)

with open('../overcooked_ai_py/data/human_data/clean_test_trials.pickle', 'rb') as f:
    dfB = pickle.load(f)
    df = dfA.append(dfB, ignore_index=True)

df = df[df['layout_name'] == 'asymmetric_advantages']
print(df)


env = OverCookedEnv(scenario="asymmetric_advantages", episode_length=500)
replay_buffer = ReplayBuffer(
    obs_shape=(2, 10056),
    action_shape=(2, 1),
    reward_shape=(2, 1),
    dones_shape=(2, 1),
    device='cpu',
    capacity=20385
)

for i, data in tqdm(df.iterrows()):
    state = data['state'].replace('\'', '\"').replace("False", 'false').replace("True", 'true')

    state_dict = json.loads(state)
    from pprint import pprint
    state = OvercookedState.from_dict(state_dict)
    state._all_orders = [Recipe(('onion', 'onion', 'onion'))]

    action = data['joint_action']
    action = json.loads(action.replace('\'', '\"').lower())

    action[0] = tuple(action[0]) if type(action[0]) == list else action[0]
    action[1] = tuple(action[1]) if type(action[1]) == list else action[1]

    next_state, mdp_infos = env.overcooked.mdp.get_state_transition(state, joint_action=action,
                                            display_phi=None,
                                            motion_planner=env.overcooked.mp)

    current_feature = np.array(env.overcooked.featurize_state_mdp(state)).reshape(2, -1)

    action = np.array(_convert_action_to_int(action)).reshape(-1, 1)
    reward = np.array(mdp_infos['shaped_reward_by_agent']).reshape(-1, 1)
    next_feature = np.array(env.overcooked.featurize_state_mdp(next_state)).reshape(2, -1)
    dones = np.array([0, 0]).reshape(-1, 1)

    replay_buffer.add(current_feature, action, reward, next_feature, dones)

replay_buffer.save('.', 20385)



