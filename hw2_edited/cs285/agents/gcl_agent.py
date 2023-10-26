import numpy as np

from .base_agent import BaseAgent
from cs285.policies.CostNN import CostNN
from cs285.infrastructure.replay_buffer import ReplayBuffer


class GCLAgent(BaseAgent):
    def __init__(self, env, agent_params, reward_params):
        super(GCLAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.reward_params = reward_params
        # self.expert_params = expert_params
        # self.gamma = self.agent_params['gamma']
        # self.standardize_advantages = self.agent_params['standardize_advantages']
        # self.nn_baseline = self.agent_params['nn_baseline']
        # self.reward_to_go = self.agent_params['reward_to_go']
        # self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.cost = CostNN(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.reward_params['n_layers'],
            self.reward_params['size'],
            learning_rate=self.reward_params['learning_rate'],
            action_shape = self.reward_params['action_shape']
        )

        # replay buffer
        self.expert_replay_buffer = ReplayBuffer(1000000)

    def train(self, costs_samp, costs_demo, probs):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """
        train_log = self.cost.update(costs_samp, costs_demo, probs)
        return train_log


    #####################################################
    #####################################################

    def add_to_expert_replay_buffer(self, paths):
        self.expert_replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.expert_replay_buffer.sample_random_data(batch_size)

    def save(self, path):
        return self.cost.save(path)

    def sample_expert(self, batch_size):
        return self.expert_replay_buffer.sample_recent_data(batch_size, concat_rew=False)
