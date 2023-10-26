from collections import OrderedDict
import pickle
import os
import sys
import time

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu

from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.action_noise_wrapper import ActionNoiseWrapper

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below


class GCL_Trainer(object):

    def __init__(self, params):
        # print(params)
        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        self.env = gym.make(self.params['env_name'])
        self.env.seed(seed)

        # Add noise wrapper
        if params['action_noise_std'] > 0:
            self.env = ActionNoiseWrapper(self.env, seed, params['action_noise_std'])

        # import plotting (locally if 'obstacles' env)
        if not(self.params['env_name']=='obstacles-cs285-v0'):
            import matplotlib
            matplotlib.use('Agg')

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes

        self.ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        self.ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = self.ac_dim
        self.params['agent_params']['ob_dim'] = self.ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10


        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

        #############
        ## REWARD
        #############
        gcl_class = self.params['gcl_class']
        self.gcl_agent = gcl_class(self.env, self.params['agent_params'], self.params['reward_params'])
        self.action_shape = self.params['reward_params']['action_shape']

        model = MLPPolicyPG(self.ac_dim, self.ob_dim,2,64,discrete=discrete, learning_rate=5e-3, nn_baseline=False)
        checkpoint = torch.load('/home/oyindamola/Research/homework_fall2021/hw2/data/q2_pg_q1_sb_no_rtg_dsa_CartPole-v0_23-03-2022_13-45-39/1/agent_itr_91.pt')
        model.load_state_dict(checkpoint)

        self.expert_policy = model
        self.is_gcl = False


    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """


        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()
        expert_ntraj = 100

        # experts_path = utils.sample_n_trajectories(self.env, self.expert_policy, expert_ntraj, self.params['ep_len'], render=False, render_mode=('rgb_array'))
        # print(experts_path[0])
        # for i in range(expert_ntraj):
        #     experts_path[i]['traj_probs'] = np.ones(experts_path[i]['traj_probs'].shape)
        # print(experts_path[0])
        experts_path = np.load('./pg_cartpole_100.npy', allow_pickle=True)
        # np.save('./pg_cartpole_100', experts_path)
        # experts_path = utils.convert_expertlistofrollouts(experts_path)
        self.gcl_agent.add_to_expert_replay_buffer(experts_path)



        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.log_metrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False



            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(itr,
                                initial_expertdata, collect_policy,
                                self.params['batch_size'],is_gcl=self.is_gcl,reward_class= self.gcl_agent.cost)


            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            #reward training


            # train reward (using sampled data from replay buffer and expert actions)
            reward_train_logs = self.train_reward()
            # reward_train_logs = []
            # train agent (using sampled data from replay buffer)
            train_logs = self.train_agent()

            # log/save
            if self.log_video or self.log_metrics:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, eval_policy, train_video_paths, train_logs, reward_train_logs)

                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))
                    self.gcl_agent.save('{}/cost_itr_{}.pt'.format(self.params['logdir'], itr))

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, initial_expertdata, collect_policy, batch_size, is_gcl = False, reward_class = None):
        # TODO: GETTHIS from HW1
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        # TODO decide whether to load training data or use the current policy to collect more data
        # HINT: depending on if it's the first iteration or not, decide whether to either
                # (1) load the data. In this case you can directly return as follows
                # ``` return loaded_paths, 0, None ```

                # (2) collect `self.params['batch_size']` transitions

        # TODO collect `batch_size` samples to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']
        print("\nCollecting data to be used for training...")
        #
        # if itr == 0:
        #     utils.sample_expert(self.env, collect_policy, batch_size, self.params['ep_len'])
        #
        #


        paths, envsteps_this_batch = utils.sample_trajectories(self.env, collect_policy, batch_size, self.params['ep_len'], is_gcl, reward_class , render=False, render_mode=('rgb_array'))

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True,render_mode=('rgb_array'))

        return paths, envsteps_this_batch, train_video_paths


    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            # TODO sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch,probs_batch = self.agent.sample(self.params['train_batch_size'])
            # print("Obs",ob_batch.shape)
            # TODO use the sampled data to train an agent
            # HINT: use the agent's train function
            # HINT: keep the agent's training log for debugging

            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs
        print('Done Training agent !!!...')
        return all_logs

    def train_reward(self):
        print('\nTraining reward using sampled data from replay buffer and expert data...')
        all_logs = []
        for train_step in range(self.params['num_reward_train_steps_per_iter']):

            # TODO sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch, probs_batch = self.agent.sample(self.params['train_batch_size'])

            e_ob_batch, e_ac_batch, e_re_batch, e_next_ob_batch, e_terminal_batch, e_probs_batch = self.gcl_agent.sample(self.params['train_batch_size'])

            obs_concat = np.concatenate((e_ob_batch, ob_batch), axis = 0)
            acs_concat = np.concatenate((e_ac_batch, ac_batch), axis = 0)
            probs_concat = np.concatenate((e_probs_batch, probs_batch), axis = 0)

            states = ptu.from_numpy(obs_concat)
            probs = ptu.from_numpy(probs_concat)
            actions = ptu.from_numpy(acs_concat)
            states_expert = ptu.from_numpy(e_ob_batch)
            actions_expert = ptu.from_numpy(e_ac_batch)


            costs_samp = self.gcl_agent.cost(torch.cat((states, actions.reshape(-1, self.action_shape)), dim=-1))
            costs_demo = self.gcl_agent.cost(torch.cat((states_expert, actions_expert.reshape(-1, self.action_shape)), dim=-1))


            train_log = self.gcl_agent.train(costs_samp, costs_demo,probs)
            all_logs.append(train_log)
        print("Done Training Reward !!!")
        return all_logs
    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs, reward_all_logs):

        last_log = all_logs[-1]
        reward_last_logs = reward_all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'], is_gcl=True, reward_class=self.gcl_agent.cost)

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]


            # train_reward_model = [(-self.gcl_agent.cost(torch.cat((path['observation'], path['action'].reshape(-1, self.ac_dim)), dim=-1))).sum() for path in paths]


            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Iteration"] = itr
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)


            # logs["Train_AverageReturnModel"] = np.mean(train_reward_model)
            # logs["Train_StdReturnModel"] = np.std(train_reward_model)
            # logs["Train_MaxReturnModel"] = np.max(train_reward_model)
            # logs["Train_MinReturnModel"] = np.min(train_reward_model)
            # logs["Train_AverageEpLenModel"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            logs["RewardTrainingLoss"] = reward_last_logs['TrainingLoss']
            logs["CostDemo"] = reward_last_logs['CostDemo']
            logs["CostSamples"] = reward_last_logs['CostSamples']

            if itr == 0:
                self.initial_return = np.mean(train_returns)
                # self.initial_return_model = np.mean(train_reward_model)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return
            # logs["Initial_DataCollection_AverageReturnModel"] = self.initial_return_model

            # perform the logging
            for key, value in logs.items():
                # print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)



            self.logger.dump_tabular(itr, logs)
            print('Done logging...\n\n')

            self.logger.flush()
