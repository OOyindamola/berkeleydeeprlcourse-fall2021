import torch
from cs285.policies.MLP_policy import MLPPolicyAC
import gym
import pickle
import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
from cs285.infrastructure import pytorch_util as ptu

def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)

    return im

seed = 1
env_name = 'HalfCheetah-v2'
env = gym.make(env_name)
env.seed(seed)
discrete = isinstance(env.action_space, gym.spaces.Discrete)
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

model = MLPPolicyAC(ac_dim, ob_dim,2,32,discrete=discrete, learning_rate=5e-3)




returns = []
observations = []
actions = []
frames = []
num_rollouts= 2
for i in range(num_rollouts):
    state = env.reset()
    state, r, done, _ = env.step(env.action_space.sample())
    print('iter', i)
    done = False
    totalr = 0.
    steps = 0
    checkpoint = torch.load('/home/oyindamola/Research/homework_fall2021/hw3/data/q5_100_1_HalfCheetah-v2_22-02-2022_12-57-22/1/agent_itr_100.pt')
    model.load_state_dict(checkpoint)
    while not done:

        action =ptu.to_numpy(model(ptu.from_numpy(state)).sample())[0]
        print(action)
        # print(action.shape)
        # action = env.action_space.sample()

        observations.append(state)
        actions.append(action)

        frame = env.render(mode='rgb_array')

        frames.append(_label_with_episode_number(frame, episode_num=i))

        state, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if done:
            break
            print(steps)
    print(totalr)
returns.append(totalr)
env.close()

imageio.mimwrite(os.path.join('.././videos/', env_name+'-Learner-AC.gif'), frames, fps=60)
print('returns', returns)
print('mean return', np.mean(returns))
print('std of return', np.std(returns))

expert_data = {'observations': np.array(observations),
               'actions': np.array(actions),
               'mean_return': np.mean(returns),
                'std_return': np.std(returns)}


# output_dir = os.path.join(os.getcwd(), 'data')
# filename = '{}_data_{}_rollouts.pkl'.format("Ant-v2-learner-dagger", num_rollouts)
# with open(output_dir + '/' + filename,'wb') as f:
#      pickle.dump(expert_data, f)
