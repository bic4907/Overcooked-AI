import cv2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from overcooked_ai_py.env import OverCookedEnv


env = OverCookedEnv(scenario="asymmetric_advantages")
state = env.reset()

input_action = 5

for _ in range(10000):
    action = np.array([env.action_space.sample() for _ in range(2)])
    action[0] = input_action
    action[1] = 4
    input_action = 4
    next_state, reward, done, info = env.step(action=action)

    image = env.render()
    cv2.imshow('Image', image)
    print(reward, done)
    key = cv2.waitKey(0)

    ''' Print State '''
    onehot_state = env.get_onehot_state()[0]
    print(onehot_state.shape)
    width, height, channel = onehot_state.shape[1], onehot_state.shape[0], onehot_state.shape[2]
    print(width, height, channel)
    count_width = 6
    count_height = round(channel / count_width)

    axes = []
    fig = plt.figure()

    for a in range(count_height * count_width):
        b = onehot_state[:, :, a]
        b = cv2.rotate(b, 2)
        b = cv2.flip(b, 0)
        axes.append(fig.add_subplot(count_height, count_width, a + 1))
        subplot_title = (str(a))
        axes[-1].set_title(subplot_title)
        plt.imshow(b)
    fig.tight_layout()
    plt.show()

    ''' Input Action '''
    if key == ord('a'):
        input_action = 3
    elif key == ord('s'):
        input_action = 1
    elif key == ord('d'):
        input_action = 2
    elif key == ord('w'):
        input_action = 0
    elif key == ord('m'):
        input_action = 5
    else:
        input_action = 4

    if done:
        state = env.reset()
    else:
        state = next_state

