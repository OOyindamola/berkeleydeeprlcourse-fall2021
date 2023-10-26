import torch
from cs285.policies.MLP_policy import MLPPolicyPG
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
env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(seed)
discrete = isinstance(env.action_space, gym.spaces.Discrete)
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

print(env.observation_space.shape)
print(env.observation_space.shape[0] + 1)
model = MLPPolicyPG(ac_dim, ob_dim,2,64,discrete=discrete, learning_rate=5e-3, nn_baseline=False)




returns = []
observations = []
next_observations = []
actions = []
terminals = []
frames = []
rewards = []
data = {}
num_rollouts= 10000
for i in range(num_rollouts):
    state = env.reset()
    print('iter', i)
    done = False
    totalr = 0.
    steps = 0
    checkpoint = torch.load('/home/oyindamola/Research/homework_fall2021/hw2/data/q2_pg_q1_lb_rtg_na_CartPole-v0_22-02-2022_10-11-01/1/agent_itr_99.pt')
    # checkpoint = torch.load('/home/oyindamola/Research/homework_fall2021/hw2/data/q2_pg_q1_sb_no_rtg_dsa_CartPole-v0_23-03-2022_13-45-39/1/agent_itr_91.pt')
    model.load_state_dict(checkpoint)

    while not done:
        action =ptu.to_numpy(model(ptu.from_numpy(state)).sample())
        observations.append(state)

        # frame = env.render(mode='rgb_array')
        # frames.append(_label_with_episode_number(frame, episode_num=i))
        state, r, done, _ = env.step(action)
        totalr += r
        steps += 1

        actions.append(action)
        next_observations.append(state)
        terminals.append(int(done))
        rewards.append(r)

        if done:
            print(steps,totalr)
            break

    print(totalr)

expert_data = {'observations': np.array(observations),
               'actions': np.array(actions),
               'next_observations': np.array(next_observations),
               'terminals' : np.array(terminals),
               'rewards': np.array(rewards)}


np.save('./pg_cartpole_expert_10000', expert_data)
print(expert_data['observations'].shape)
print(expert_data['actions'].shape)
returns.append(totalr)
env.close()

# imageio.mimwrite(os.path.join('.././videos/', env_name+'-Learner-AC.gif'), frames, fps=60)
# print('returns', returns)
# print('mean return', np.mean(returns))
# print('std of return', np.std(returns))
#
# expert_data = {'observations': np.array(observations),
#                'actions': np.array(actions),
#                'mean_return': np.mean(returns),
#                 'std_return': np.std(returns)}


# output_dir = os.path.join(os.getcwd(), 'data')
# filename = '{}_data_{}_rollouts.pkl'.format("Ant-v2-learner-dagger", num_rollouts)
# with open(output_dir + '/' + filename,'wb') as f:
#      pickle.dump(expert_data, f)
