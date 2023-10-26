import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions
import tensorflow as tf
from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy

EPS = 1e-8
def mlp(x,
        hidden_layers,
        activation=nn.Tanh,
        size=2,
        output_activation=nn.Identity):
    """
        Multi-layer perceptron
    """
    net_layers = []

    if len(hidden_layers[:-1]) < size:
        hidden_layers[:-1] *= size

    for size in hidden_layers[:-1]:
        layer = nn.Linear(x, size)
        net_layers.append(layer)

        # For discriminator
        if activation.__name__ == 'ReLU':
            net_layers.append(activation(inplace=True))
        elif activation.__name__ == 'LeakyReLU':
            net_layers.append(activation(.2, inplace=True))
        else:
            net_layers.append(activation())
        x = size

    net_layers.append(nn.Linear(x, hidden_layers[-1]))
    net_layers += [output_activation()]

    return nn.Sequential(*net_layers)
class Discriminator(nn.Module):
    """
        Disctimates between expert data and samples from
        learned policy.
        It recovers the advantage f_theta_phi  used in training the
        policy.

        The Discriminator has:

        g_theta(s): A state-only dependant reward function
            This allows extraction of rewards that are disentangled
            from the dynamics of the environment in which they were trained

        h_phi(s): The shaping term
                  Mitigates unwanted shaping on the reward term g_theta(s)

        f_theta_phi = g_theta(s) + gamma * h_phi(s') - h_phi(s)
        (Essentially an advantage estimate)
    """
    def __init__(self, obs_dim, gamma, disc_params):
        super(Discriminator, self).__init__()

        self.gamma = gamma

        # *g(s) = *r(s) + const
        #  g(s) recovers the optimal reward function +  c
        # self.g_theta =  ptu.build_mlp(input_size=obs_dim,
        #                                output_size=1,
        #                                n_layers=disc_params['disc_g_n_layers'],
        #                                size=disc_params['disc_g_size'],
        #                                activation=disc_params['g_activation'] )
        #
        # # *h(s) = *V(s) + const (Recovers optimal value function + c)
        # self.h_phi =ptu.build_mlp(input_size=obs_dim,
        #                                output_size=1,
        #                                n_layers=disc_params['disc_h_n_layers'],
        #                                size=disc_params['disc_h_size'],
        #                                activation=disc_params['h_activation'] )

        self.g_theta = mlp(obs_dim, [32, 1], nn.Identity, size = 1)

        # *h(s) = *V(s) + const (Recovers optimal value function + c)
        self.h_phi = mlp(obs_dim, [32, 32, 1], nn.LeakyReLU, size = 2)

        self.g_theta.to(ptu.device)
        self.h_phi.to(ptu.device)
        self.sigmoid = nn.Sigmoid()



    def forward(self, obs, obs_n, dones ):
        """
            Returns the estimated reward function / Advantage
            estimate. Given by:

            f(s, a, s') = g(s) + gamma * h(s') - h(s)


            Parameters
            ----------
            data    | [obs, obs_n, dones] #observation, next_observation, terminals
        """
        # print("In discriminator")
        # obs, obs_n, dones = data
        g_s = torch.squeeze(self.g_theta(obs), axis=-1)

        shaping_term = self.gamma * \
            (1 - dones) * self.h_phi(obs_n).squeeze() - \
            self.h_phi(obs).squeeze(-1)

        f_thet_phi = g_s + shaping_term

        return f_thet_phi

    def discr_value(self, log_p,   obs, obs_n, dones):
        """
            Calculates the disctiminator output
                D = exp(f(s, a, s')) / [exp(f(s, a, s')) + pi(a|s)]
        """
        # print("Re discriminator")
        adv = self( obs, obs_n, dones)



        exp_adv = torch.exp(adv)
        value = exp_adv / (exp_adv + torch.exp(log_p) + EPS)
        # value2 = adv / (adv + log_p + EPS)

        # print("adv", exp_adv.shape, torch.exp(log_p).shape, (exp_adv + torch.exp(log_p) + EPS).shape)
        # print("adv", self.sigmoid(value).shape, log_p.shape)
        return self.sigmoid(value)


class MLPActor(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 agent_params,
                 disc_params,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.agent_params = agent_params
        self.disc_params = disc_params
        self.ac_dim = agent_params['ac_dim']
        self.ob_dim =  agent_params['ob_dim']
        self.n_layers = agent_params['n_layers']
        self.discrete = agent_params['discrete']
        self.size = agent_params['size']
        self.learning_rate = agent_params['learning_rate']
        self.activation = agent_params['activation']
        self.training = training
        self.nn_baseline = nn_baseline

        self.gamma = agent_params['gamma']
        self.disc = Discriminator(self.ob_dim, self.gamma, disc_params)

        self.disc_optimizer = optim.Adam(self.disc.parameters(),
                                    disc_params['disc_g_learning_rate'],
                                    betas=[.5, .9])

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size,
                                           activation=self.activation )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate,
                                        betas=[.5, .9])
        else:
            self.logits_na = None
            # self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
            #                           output_size=self.ac_dim,
            #                           n_layers=self.n_layers, size=self.size)

            self.mean_net = mlp(self.ob_dim,
                              [64, 64] + [self.ac_dim],
                              nn.Tanh,
                              size=2)

            log_std = -.5 * np.ones(self.ac_dim, dtype=np.float32)
            self.logstd = nn.Parameter(ptu.from_numpy(log_std))
            # self.logstd = nn.Parameter(
            #     torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            # )
            # print(self.logstd.shape)
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate,
                betas=[.5, .9]
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def save_disc(self, filepath):
        torch.save(self.disc.state_dict(), filepath)

    def sample_action(self, obs):
        pi = self.forward(obs)
        action = pi.sample()
        if self.discrete:
            log_prob = pi.log_prob(action)
        else:
            log_prob = pi.log_prob(action).sum(
                axis=-1)
        #prob = torch.exp(log_prob).unsqueeze(0)

        # action.requires_grad = True
        return action, log_prob

    ###########################################
    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs):
        with torch.no_grad():
            if len(obs.shape) > 1:
                observation = obs
            else:
                observation = obs[None]

            observation = ptu.from_numpy(observation)
            pi = self.forward(observation)
            action = pi.sample()
            if self.discrete:
                log_prob = pi.log_prob(action)
            else:
                log_prob = pi.log_prob(action).sum(
                    axis=-1)
                # print("logprob: ", log_prob)
            # action, log_prob = self.sample_action(observation)

            # TODO return the action that the policy prescribes
        return ptu.to_numpy(action), ptu.to_numpy(log_prob)  #action

    def sample_policy(self, obs, action):
        pi = self.forward(obs)
        # action = dist.sample()
        if self.discrete:
            log_prob = pi.log_prob(action)
        else:
            log_prob = pi.log_prob(action).sum(
                axis=-1)
        #prob = torch.exp(log_prob).unsqueeze(0)

        # action.requires_grad = True

        return pi, log_prob



    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    def log_p(self, pi, a):
        return pi.log_prob(a).sum(
            axis=-1)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)

            scale_tril = torch.exp(self.logstd)

            batch_dim = batch_mean.shape[0]
            # scale_tril = scale_tril.to(ptu.device)


            # batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.Normal(
                loc=batch_mean,
                scale=scale_tril
            )
            return action_distribution
            # batch_mean = self.mean_net(observation)
            # scale_tril = torch.diag(torch.exp(self.logstd))
            # batch_dim = batch_mean.shape[0]
            # batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            # action_distribution = distributions.MultivariateNormal(
            #     batch_mean,
            #     scale_tril=batch_scale_tril,
            # )
            # return action_distribution

#####################################################
#####################################################

class MLPActorPPO(MLPActor):
    def __init__(self, agent_params,disc_params,**kwargs):

        super().__init__(agent_params,disc_params,**kwargs)
        self.disc_loss = nn.BCELoss()


    def compute_pi_loss(self, exp_data, act, log_p):
        """
            Pi loss

            obs_b, obs_n_b, dones_b are expert demonstrations. They
            will be used in finding `adv_b` - the Advantage estimate
            from the learned reward function
        """
        e_obs, e_n_obs, e_dones = exp_data

        log_p_old = log_p

        e_obs = ptu.from_numpy(e_obs).type(torch.float32)
        e_n_obs = ptu.from_numpy(e_n_obs).type(torch.float32)
        act = ptu.from_numpy(act).type(torch.float32)
        e_dones = ptu.from_numpy(e_dones).type(torch.float32)
        log_p_old = ptu.from_numpy(log_p_old).type(torch.float32)

        clip_ratio = self.agent_params['clip_ratio']
        #
        # print("obs: ", e_obs)
        # print("act_b: ", act)
        # returns new_pi_normal_distribution, logp_act
        pi_new, log_p_ = self.sample_policy(e_obs, act)
        log_p_ = log_p_.type(torch.float32)  # From torch.float64

        # print("act: ", act)
        # print("pi: ", pi_new, log_p_ )
        # Predict adv using learned reward function

        # r_t^(s, a) = f(s, a) - log pi(a|s) - log pi(a|s) is entropy of pi
        # r_t^(s, a) = A(s, a)

        adv_b = self.disc(e_obs, e_n_obs, e_dones) - log_p_old
        # print("adv_b: ", adv_b)

        adv_b = (adv_b - adv_b.mean()) / adv_b.std()

        pi_diff = log_p_ - log_p_old

        pi_ratio = torch.exp(pi_diff)

        # Soft PPO update - Encourages entropy in the policy

        # i.e. Act as randomly as possibly while maximizing the objective
        # Example case: pi might learn to take a certain action for a given
        # state every time because it has some good reward, but forgo
        # trying other actions which might have higher reward

        # A_old_pi(s, a) = A(s, a) - entropy_reg * log pi_old(a|s)
        entropy_reg = self.agent_params['entropy_reg']
        clip_ratio = self.agent_params['clip_ratio']
        # print(entropy_reg,clip_ratio)
        adv_b = adv_b - (  entropy_reg * log_p_old)

        min_adv = torch.where(adv_b >= 0, (1 + clip_ratio) * adv_b,
                              (1 - clip_ratio) * adv_b)

        pi_loss = -torch.mean(torch.min(pi_ratio * adv_b, min_adv))
        kl = -(pi_diff).mean().item()
        entropy = pi_new.entropy().mean().item()

        return pi_loss, kl, entropy


    def update_pi(self,exp_data, act,  log_p, kl_start):

        pi_loss_old, kl, entropy = self.compute_pi_loss(exp_data, act, log_p)

        for i in range(self.agent_params['num_agent_train_steps_per_iter']):
            # print(i)
            self.optimizer.zero_grad()

            pi_loss, kl, entropy = self.compute_pi_loss(exp_data, act, log_p)
            if kl_start and kl > 1.5 * self.agent_params[
                    'max_kl']:  # Early stop for high Kl
                print('Max kl reached: ', kl, '  iter: ', i)
                break

            pi_loss.backward()
            self.optimizer.step()

        train_log = {
            'PiLossOld': ptu.to_numpy(pi_loss_old).item(),
            'PiLoss': ptu.to_numpy(pi_loss).item(),
            'PiLKl': kl,
            'Entropy': entropy,
        }

        return train_log


    def compute_disc_loss(self, observations, next_observation, terminals, log_p, label):
        """
            Disciminator loss

            log D_theta_phi (s, a, s') − log(1 − D_theta_phi (s, a, s')) ... (1)

            (Minimize likelohood of policy samples while increase likelihood
            of expert demonstrations)

            D_theta_phi = exp(f(s, a, s')) / [exp(f(s, a, s')) + pi(a|s)]

            Substitute this in eq (1):
                        = f(s, a, s') - log p(a|s)


            Args:
                traj: (s, a, s') samples
                label: Label for expert data or pi samples
        """



        observations = ptu.from_numpy(observations).type(torch.float32)
        next_observation = ptu.from_numpy(next_observation).type(torch.float32)
        terminals = ptu.from_numpy(terminals).type(torch.float32)
        log_p = ptu.from_numpy(log_p).type(torch.float32)
        label = ptu.to_gpu(label)

        disc_x = self.disc.discr_value(log_p,  observations,next_observation, terminals).view(-1)

        # print(disc_x.shape, label.shape)
        loss = self.disc_loss(disc_x, label)


        return disc_x, loss

    def update_disc(self, exp_data, demo_data, log_p):

        e_obs, e_n_obs, e_dones = exp_data
        print("Done: ", np.count_nonzero(e_dones))
        d_obs = demo_data['observations']
        d_n_obs = demo_data['next_observations']
        d_dones = demo_data['terminals']

        real_label= self.disc_params['real_label']
        pi_label= self.disc_params['pi_label']
        print(log_p)
        label = torch.full((self.disc_params['batch_size'], ), real_label, dtype=torch.float32)
        # print("log p sum:", log_p.sum())
        # print("non zeros p sum:",  np.count_nonzero(log_p))
        demo_info = self.compute_disc_loss(e_obs, e_n_obs, e_dones, log_p=log_p, label=label)
        pi_samples_info = self.compute_disc_loss(d_obs, d_n_obs, d_dones,
                                            log_p=log_p,
                                            label=label.fill_(pi_label))

        _, err_demo_old = demo_info
        _, err_pi_samples_old = pi_samples_info

        disc_loss_old = (err_demo_old + err_pi_samples_old).mean().item()
        # print("err_demo_old: ", err_demo_old.mean())
        # print("err_pi_samples_old: ", err_pi_samples_old.mean())
        # print("discc: ", disc_loss_old)

        for i in range(self.disc_params['num_disc_train_steps_per_iter']):
            # Train with expert demonstrations
            # log(D(s, a, s'))


            self.disc.zero_grad()

            av_demo_output, err_demo = self.compute_disc_loss(e_obs, e_n_obs,
                            e_dones, log_p=log_p, label=label.fill_(self.disc_params['real_label']))

            err_demo.backward()
            # works too, but compute backprop once
            # See "disc_loss_update_test.ipynb"

            # Train with policy samples
            # - log(D(s, a, s'))
            label.fill_(pi_label)
            av_pi_output, err_pi_samples =  self.compute_disc_loss(d_obs, d_n_obs, d_dones,
                                                log_p=log_p,
                                                label=label)

            err_pi_samples = -err_pi_samples
            err_pi_samples.backward()
            loss = err_demo + err_pi_samples

            # - To turn minimization to Maximization of the objective
            #-loss.backward()

            self.disc_optimizer.step()
        print("Loss: ", loss)
        train_log = {
            'AdvantagePi': ptu.to_numpy(av_pi_output).mean().item(),
            'AdvantageDemo': ptu.to_numpy(av_demo_output).mean().item(),
            'Disc_PiTrainingLoss': ptu.to_numpy(err_pi_samples).item(),
            'Disc_DemoTrainingLoss': ptu.to_numpy(err_demo).item(),
            'DiscTrainingLossOld': disc_loss_old,
            'DiscTrainingLoss': loss

        }

        return train_log
