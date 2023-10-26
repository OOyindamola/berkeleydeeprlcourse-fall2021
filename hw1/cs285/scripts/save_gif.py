import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt

import gym
def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)

    return im


def save_random_agent_gif(env):
    returns = []
    observations = []
    actions = []
    frames = []
    for i in range(5):
        state = env.reset()
        print('iter', i)
        done = False
        totalr = 0.
        steps = 0

        for t in range(500):
            action = env.action_space.sample()

            frame = env.render(mode='rgb_array')
            frames.append(_label_with_episode_number(frame, episode_num=i))

            state, _, done, _ = env.step(action)
            if done:
                break

    env.close()

    imageio.mimwrite(os.path.join('./videos/', 'ant-v2-expert.gif'), frames, fps=60)


env = gym.make('Ant-v2')
save_random_agent_gif(env)
