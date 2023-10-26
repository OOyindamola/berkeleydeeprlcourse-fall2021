import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy
import pickle
import gym
def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)
    # print(im)
    return im


def save_random_agent_gif(env):
    loaded_expert_policy = LoadedGaussianPolicy('../policies/experts/Ant.pkl')
    returns = []
    observations = []
    actions = []
    frames = []
    num_rollouts= 1
    for i in range(num_rollouts):
        state = env.reset()
        print('iter', i)
        done = False
        totalr = 0.
        steps = 0

        for t in range(250):
            action = loaded_expert_policy.get_action(state[None,:])
            # action= env.action_space.sample()
            observations.append(state)
            actions.append(action)

            frame = env.render(mode='rgb_array')
            frames.append(_label_with_episode_number(frame, episode_num=i))

            state, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if done:
                break
    returns.append(totalr)
    env.close()

    imageio.mimwrite(os.path.join('./videos/', 'ant-v2-expert.gif'), frames, fps=60)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions),
                   'mean_return': np.mean(returns),
                    'std_return': np.std(returns)}


    # output_dir = os.path.join(os.getcwd(), 'data')
    # filename = '{}_data_{}_rollouts.pkl'.format("Ant-v2", num_rollouts)
    # with open(output_dir + '/' + filename,'wb') as f:
    #      pickle.dump(expert_data, f)



env = gym.make('Ant-v2')
save_random_agent_gif(env)
