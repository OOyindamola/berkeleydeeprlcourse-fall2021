from cs285.infrastructure.utils import *

class ExpertBuffer():
    """
        Expert demonstrations Buffer

        Args:
            path (str): Path to the expert data file

    Loading the data works with files written using:
        np.savez(file_path, **{key: value})

    So the loaded file can be accessed again by key
        value = np.load(file_path)[key]

        data format: { 'iteration_count': transitions }
            Where transitions is a key,value pair of
            expert data samples of size(-1)  `steps_per_epoch`
    """
    def __init__(self, path):
        data_file = np.load(path, allow_pickle=True)
        self.load_rollouts(data_file)

        self.ptr = 0

    def load_rollouts(self, data_file):
        """
            Convert a list of rollout dictionaries into
            separate arrays concatenated across the arrays
            rollouts
        """
        #
        # self.obs = data_file.item()['observations']
        # self.obs_n = data_file.item()['next_observations']
        # self.dones = data_file.item()['terminals']
        # self.rewards = data_file.item()['rewards']
        #
        # self.size = self.dones.shape[0]


        # get all iterations transitions
        data = [traj for traj in data_file.values()]

        try:
            #  traj in x batch arrays. Unroll to 1
            data = np.concatenate(data)

            # Handle differnce in saving formats
        except ValueError:
            # Zero dimension array can't be
            data = [d[None] for d in data]
            data = np.concatenate(data)

        self.obs = np.concatenate([path['observation'] for path in data])
        self.obs_n = np.concatenate(
            [path['next_observation'] for path in data])
        self.dones = np.concatenate([path['terminal'] for path in data])

        self.size = self.dones.shape[0]

    def get_random(self, batch_size):
        """
            Fetch random expert demonstrations of size `batch_size`

            Returns:
            obs, obs_n, dones
        """
        idx = np.random.randint(self.size, size=batch_size)
        
        return (
            self.obs[idx],
            # self.act[idx],
            self.obs_n[idx],
            self.dones[idx])

    def get(self, batch_size):
        """
            Samples expert trajectories by order
            of saved iterations

            Returns:
            obs, obs_n, dones
        """
        if self.ptr + batch_size > self.size:
            self.ptr = 0

        idx = slice(self.ptr, self.ptr + batch_size)

        self.ptr = ((self.ptr + 1) * batch_size) % self.size

        return (
            self.obs[idx],
            # self.act[idx],
            self.obs_n[idx],
            self.dones[idx])


class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.paths = []
        self.obs = None
        self.acs = None
        self.concatenated_rews = None
        self.unconcatenated_rews = None
        self.next_obs = None
        self.terminals = None
        self.traj_probs = None

    def add_rollouts(self, paths, noised=False):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, next_observations, terminals, concatenated_rews, unconcatenated_rews, traj_probs = convert_listofrollouts(paths)

        if noised:
            observations = add_noise(observations)
            next_observations = add_noise(next_observations)

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.concatenated_rews = concatenated_rews[-self.max_size:]
            self.unconcatenated_rews = unconcatenated_rews[-self.max_size:]
            self.traj_probs = traj_probs[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]
            self.concatenated_rews = np.concatenate(
                [self.concatenated_rews, concatenated_rews]
            )[-self.max_size:]
            if isinstance(unconcatenated_rews, list):
                self.unconcatenated_rews += unconcatenated_rews  # TODO keep only latest max_size around
            else:
                self.unconcatenated_rews.append(unconcatenated_rews)  # TODO keep only latest max_size around
            self.traj_probs = np.concatenate([self.traj_probs, traj_probs])[-self.max_size:]
    ########################################
    ########################################

    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return self.paths[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1):
        return self.paths[-num_rollouts:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):

        assert self.obs.shape[0] == self.acs.shape[0] == self.concatenated_rews.shape[0] == self.next_obs.shape[0] == self.terminals.shape[0] == self.traj_probs.shape[0]
        rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return self.obs[rand_indices], self.acs[rand_indices], self.concatenated_rews[rand_indices], self.next_obs[rand_indices], self.terminals[rand_indices], self.traj_probs[rand_indices]

    def sample_recent_data(self, batch_size=1, concat_rew=True):

        if concat_rew:

            return self.obs[-batch_size:], self.acs[-batch_size:], self.concatenated_rews[-batch_size:], self.next_obs[-batch_size:], self.terminals[-batch_size:],  self.traj_probs[-batch_size:]
        else:
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self.paths[index]
                index -=1
                num_recent_rollouts_to_return +=1
                num_datapoints_so_far += get_pathlength(recent_rollout)

            rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
            observations, actions, next_observations, terminals, concatenated_rews, unconcatenated_rews,traj_probs = convert_listofrollouts(rollouts_to_return)
            return observations, actions, unconcatenated_rews, next_observations, terminals, traj_probs
