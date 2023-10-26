#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20
Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
import os
# import imageio
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
from gym import wrappers
# from PIL import Image
# import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy

def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)
    print(im)
    return im


def generate_all_rollout_data():
    generate_rollout_data('../policies/experts/Ant.pkl', 'Ant-v2', 250, False, 'data')
    generate_rollout_data('../policies/experts/HalfCheetah.pkl', 'HalfCheetah-v2', 10, False, 'data')
    generate_rollout_data('../policies/experts/Hopper.pkl', 'Hopper-v2', 10, False, 'data')
    generate_rollout_data('../policies/experts/Humanoid.pkl', 'Humanoid-v2', 250, False, 'data')
    # generate_rollout_data('../policies/experts/Reacher.pkl', 'Reacher-v2', 250, False, 'data')
    generate_rollout_data('../policies/experts/Walker2d.pkl', 'Walker2d-v2', 10, False, 'data')


def generate_rollout_data(expert_policy_file, env_name, num_rollouts, render, output_dir=None, save=False, max_timesteps=None):
    print('loading and building expert policy')
    # policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')
    loaded_expert_policy = LoadedGaussianPolicy(expert_policy_file)

    with tf.compat.v1.Session():
        tf_util.initialize()

        env = gym.make(env_name)
        max_steps = max_timesteps or env.spec.max_episode_steps
        print(max_steps)

        if save:
            expert_results_dir = os.path.join(os.getcwd(), 'results', env_name, 'expert')
            env = wrappers.Monitor(env, expert_results_dir, force=True)


        expert_data = []
        returns = []

        for i in range(num_rollouts):
            print(env_name, i)
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            rollout_done=False

            observations = []
            actions = []
            frames = []
            terminals = []
            image_obs=[]
            rewards = []
            next_obs= []
            while not done:
                action = loaded_expert_policy.get_action(obs)
                # print(action[0].shape)
                # print(obs.shape)
                observations.append(obs)
                actions.append(action[0])


                obs, r, done, _ = env.step(action[0])
                totalr += r
                steps += 1
                next_obs.append(obs)
                rewards.append(r)
                if render:
                    env.render()
                    # frame = env.render(mode='rgb_array')
                    # frames.append(_label_with_episode_number(frame, episode_num=i))
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))


                if done or steps == max_steps:
                    rollout_done = True # HINT: this is either 0 or 1

                terminals.append(rollout_done)
                if rollout_done:
                    break

            returns.append(totalr)

            env.close()

            expert_data.append({"observation" : np.array(observations, dtype=np.float32),
                    "image_obs" : np.array(image_obs, dtype=np.uint8),
                    "reward" : np.array(rewards, dtype=np.float32),
                    "action" : np.array(actions, dtype=np.float32),
                    "next_observation": np.array(next_obs, dtype=np.float32),
                    "terminal": np.array(terminals, dtype=np.float32)})


            # imageio.mimwrite(os.path.join('./videos/', 'ant-v2-experts.gif'), frames, fps=60)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))



        if output_dir !='None':
            output_dir = os.path.join(os.getcwd(), output_dir)
            filename = '{}_data_{}_rollouts.pkl'.format(env_name, num_rollouts)
            with open(output_dir + '/' + filename,'wb') as f:
                 pickle.dump(expert_data, f)

if __name__ == '__main__':
    generate_all_rollout_data()
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('expert_policy_file', type=str)
    # parser.add_argument('envname', type=str)
    # parser.add_argument("--max_timesteps", type=int)
    # parser.add_argument('--num_rollouts', type=int, default=20,
    #                     help='Number of expert roll outs')
    # parser.add_argument('--render', action='store_true')
    # parser.add_argument("--output_dir", type=str, default='data')
    # args = parser.parse_args()

    # generate_rollout_data('../policies/experts/Ant.pkl', 'Ant-v2', 10, False, 'data')

    # generate_rollout_data(args.expert_policy_file, args.envname, args.num_rollouts, args.render, args.output_dir, True, args.max_timesteps)
