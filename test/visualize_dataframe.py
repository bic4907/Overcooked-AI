import pickle
import json
import numpy as np
import pygame
import cv2
import copy

from overcooked_ai_py.env import OverCookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
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


visualizer = StateVisualizer()

for i, data in df.iterrows():
    state = data['state'].replace('\'', '\"').replace("False", 'false').replace("True", 'true')

    state_dict = json.loads(state)

    from pprint import pprint
    state = OvercookedState.from_dict(state_dict)

    state._all_orders = [Recipe(('onion', 'onion', 'onion'))] # { "ingredients" : ["onion", "onion", "onion"]}]
    #
    # state.all_orders = [('onion', 'onion', 'onion')]

    action = data['joint_action']
    action = json.loads(action.replace('\'', '\"').lower())
    print(state)
    image = visualizer.render_state(state=state, grid=env.overcooked.mdp.terrain_mtx,
                                        hud_data=StateVisualizer.default_hud_data(state))

    # image = visualizer.render_state(state=state, grid=env.overcooked.mdp.terrain_mtx,
    #                                      hud_data=None)
    buffer = pygame.surfarray.array3d(image)
    image = copy.deepcopy(buffer)
    image = np.flip(np.rot90(image, 3), 1)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (528, 464))

    cv2.imshow('Display', image)
    print(i, action)
    cv2.waitKey(1)




