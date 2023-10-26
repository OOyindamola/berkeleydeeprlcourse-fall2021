import os
import time
import json
from cs285.infrastructure.airl_trainer import AIRL_Trainer
from cs285.agents.airl_agent import AIRLAgent
# from cs285.agents.airl_agent import AIRLAgent

import os.path as osp, shutil, time, atexit, os, subprocess


class AIRL_PPO_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'activation': params['activation'],
            }

        disc_computation_graph_args = {
            'disc_g_n_layers': params['disc_g_n_layers'],
            'disc_g_size': params['disc_g_size'],
            'disc_g_learning_rate': params['disc_g_learning_rate'],
            'disc_h_n_layers': params['disc_h_n_layers'],
            'disc_h_size': params['disc_h_size'],
            'disc_h_learning_rate': params['disc_h_learning_rate'],
            'g_activation': params['g_activation'],
            'h_activation': params['h_activation'],
            }

        #
        # estimate_advantage_args = {
        #     'gamma': params['discount'],
        #     'standardize_advantages': not(params['dont_standardize_advantages']),
        #     'reward_to_go': params['reward_to_go'],
        #     'nn_baseline': params['nn_baseline'],
        #     'gae_lambda': params['gae_lambda'],
        # }

        ppo_args = {
                'gamma': params['discount'],
                'max_kl': params['max_kl'],
                'kl_start': params['kl_start'],
                'entropy_reg': params['entropy_reg'],
                'clip_ratio': params['clip_ratio'],
                'max_eps_len': params['max_eps_len']
        }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],

        }

        d_train_args={
        'num_disc_train_steps_per_iter': params['num_disc_train_steps_per_iter'],
            'real_label': params['real_label'],
                'pi_label': params['pi_label'],
            'batch_size': params['batch_size']
        }
        agent_params = {**computation_graph_args, **ppo_args, **train_args}

        disc_params = {**disc_computation_graph_args, **d_train_args}

        self.params = params
        self.params['agent_class'] = AIRLAgent
        self.params['agent_params'] = agent_params
        self.params['disc_params'] = disc_params

        self.params['batch_size_initial'] = self.params['batch_size']



        ################
        ## RL TRAINER
        ################

        self.airl_trainer = AIRL_Trainer(self.params)

    def run_training_loop(self):

        self.airl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy = self.airl_trainer.agent.actor,
            eval_policy = self.airl_trainer.agent.actor
            )
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)
def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--expert_policy', type=str)
    parser.add_argument('--n_iter', '-n', type=int, default=250)
    parser.add_argument('--action_shape', '-acs', type=int)

    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--nn_baseline', action='store_true')
    parser.add_argument('--gae_lambda', type=float, default=None)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=10000) #steps collected per train iteration
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=400) #steps collected per eval iteration
    parser.add_argument('--train_batch_size', '-tb', type=int, default=10000) ##steps used per gradient step



    #PPO params
    parser.add_argument('--max_kl', '-kl', type=int, default=1)
    parser.add_argument('--kl_start', '-kls', type=int, default=20)
    parser.add_argument('--entropy_reg', '-er', type=int, default=.1)
    parser.add_argument('--clip_ratio', '-cr', type=int, default=.2)
    parser.add_argument('--max_eps_len', '-max_eps_len', type=int, default=150)
    parser.add_argument('--real_label', '-rlabel', type=int, default=1)
    parser.add_argument('--pi_label', '-pilabel', type=int, default=0)


    #policy - agent
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=80)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--learning_rate', '-lr', type=float, default=2e-4)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--activation', type=str, default= 'tanh')


    #discriminator
    parser.add_argument('--num_disc_train_steps_per_iter', type=int, default=40)
    parser.add_argument('--disc_g_learning_rate', '-dglr', type=float, default=1e-4)
    parser.add_argument('--disc_g_n_layers', '-dgl', type=int, default=1)
    parser.add_argument('--disc_g_size', '-dgs', type=int, default=32)
    parser.add_argument('--g_activation', type=str, default= 'identity')

    parser.add_argument('--disc_h_learning_rate', '-dhlr', type=float, default=1e-4)
    parser.add_argument('--disc_h_n_layers', '-dhl', type=int, default=2)
    parser.add_argument('--disc_h_size', '-dhs', type=int, default=32)
    parser.add_argument('--h_activation', type=str, default= 'leaky_relu')

    parser.add_argument('--ep_len', type=int) #students shouldn't change this away from env's default
    # parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--seed', type=int,  nargs='+', default=[1])

    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=1)

    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--action_noise_std', type=float, default=0)

    args = parser.parse_args()
    seeds = args.seed
    params = vars(args)

    ########################################
    ##### EXPERT POLICY
    ##########################################


    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'airl_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)

    for i in range(len(seeds)):
        print(seeds[i])
        # convert to dictionary
        args.seed = seeds[i]
        params['seed'] = seeds[i]
        # print(params)

        # for policy gradient, we made a design decision
        # to force batch_size = train_batch_size
        # note that, to avoid confusion, you don't even have a train_batch_size argument anymore (above)
        params['train_batch_size'] = params['batch_size']





        seedlogdir = os.path.join(logdir, str(params['seed']))
        params['logdir'] = seedlogdir


        seedparams = params.copy()
        # print(seedparams)
        if not(os.path.exists(seedlogdir)):
            os.makedirs(seedlogdir)
            f = open(osp.join(seedlogdir, "log.txt"), 'w')
            atexit.register(f.close)
            print(colorize("Logging data to %s"%f.name, 'green', bold=True))

        with open(seedlogdir+"/params.json", "w") as outfile:
            json.dump(params, outfile, indent=1)
        ###################
        ### RUN TRAINING
        ###################

        trainer = AIRL_PPO_Trainer(seedparams)
        trainer.run_training_loop()


if __name__ == "__main__":
    main()
