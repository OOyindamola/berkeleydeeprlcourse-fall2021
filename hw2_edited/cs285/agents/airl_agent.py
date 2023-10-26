import numpy as np

from .base_agent import BaseAgent
from cs285.policies.airl_policy import MLPActorPPO
from cs285.infrastructure.replay_buffer import ReplayBuffer, ExpertBuffer


class AIRLAgent(BaseAgent):
    def __init__(self, env, agent_params, disc_params, expert_policy):
        super(AIRLAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.disc_params = disc_params

        # actor/policy
        self.actor = MLPActorPPO(
            self.agent_params,
            self.disc_params
        )

        # replay buffer
        self.expert_replay_buffer = ExpertBuffer(expert_policy)
        self.replay_buffer = ReplayBuffer(1000000)


    def train(self,exp_data, sample_disc_data, logp_batch, kl_start):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """
        disc_train_log = self.actor.update_disc(exp_data, sample_disc_data, logp_batch)
        act = sample_disc_data['actions']
        pi_train_log = self.actor.update_pi(exp_data, act,  logp_batch, kl_start)

        return disc_train_log, pi_train_log


    #####################################################
    #####################################################

    def add_to_expert_replay_buffer(self, paths):
        self.expert_replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=True)

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample_expert(self, batch_size):
        return self.expert_replay_buffer.sample_random_data(batch_size)

    def save(self, path):
        return self.cost.save(path)

    def sample_expert(self, batch_size):
        return self.expert_replay_buffer.sample_recent_data(batch_size, concat_rew=False)
