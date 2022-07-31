
from math import log2
import numpy as np
from math import ceil
from os.path import dirname



class Server(object):
    """Base module for Server"""

    def __init__(self, total_users, users_per_round, num_glob_iters, policy, c, beta, tau_max, alpha=0.9):

        self.num_glob_iters = num_glob_iters
        self.users = []
        self.selected_users = []
        self.users_per_round = users_per_round

        for i in range(total_users):
            user = User(user_id=i, tau_max=tau_max, c=c, alpha=alpha)
            self.users.append(user)

        self.tau_max = tau_max  # max delay time in seconds
        self.policy = policy  # scheduling policy
        if self.policy == 'CS-UCB-Q':  # update policy optimal reward
            self.mu_opt = 0
            user_groups = powerset([n for n in range(total_users)])  # all available chosen groups
            for group in user_groups:
                if group:
                    p = 1
                    for n in range(total_users):
                        if n in group:
                            p *= self.users[n].alpha
                        else:
                            p *= 1 - self.users[n].alpha
                    self.mu_opt += p * sorted([self.users[n].mu for n in group], reverse=True)[min(self.users_per_round - 1, len(group) - 1)]
        else:
            self.mu_opt = sorted([user.mu for user in self.users], reverse=True)[self.users_per_round - 1]
        self.kapa = ceil(len(self.users) / self.users_per_round)  # total rounds for first exploration
        self.sigma_pi = [0]  # cumulative regret values for policy pi
        self.fails = [0]  # cumulative counter for failed clients
        self.beta = beta

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            self.selected_users = self.select_users(glob_iter, self.users_per_round)
            self.update_reward_estimation()
            if any(self.selected_users):
                reward = sorted([user.r for user in self.selected_users])[0]  # min individual reward between selected users
            else:
                reward = 0
            self.sigma_pi.append(self.sigma_pi[-1] + self.mu_opt - reward)
        self.save_results()

    def select_users(self, glob_iter, users_per_round):
        if glob_iter < self.kapa and self.policy != 'CS-UCB-Q':
            return self.users[glob_iter * self.users_per_round:min((glob_iter + 1) * self.users_per_round, len(self.users))]

        np.random.seed(glob_iter)  # fix the list of user consistent

        if self.policy == 'Random':
            return np.random.choice(self.users, users_per_round, replace=False)

        elif self.policy == 'CS-UCB':
            return sorted(self.users,
                          key=lambda x: x.y_t + np.sqrt(
                              (self.users_per_round + 1) * np.log(glob_iter + 1) / x.z_t),
                          reverse=True)[:self.users_per_round]

        elif self.policy == 'CS-UCB-Q':
            available_users = []
            for user in self.users:
                if user.z_t > 0:
                    user.y_hat_t = min(user.y_t + np.sqrt(2 * np.log(glob_iter + 1) / user.z_t), 1)
                user.update_queue()

                if user.check_availability():
                    available_users.append(user)

                user.b_t = 0

            return sorted(self.users,
                          key=lambda x:
                          (1 - self.beta) * x.y_hat_t + self.beta * x.virtual_queue, reverse=True)[:min(self.users_per_round, len(available_users))]

    def update_reward_estimation(self):
        fails = 0
        for i, user in enumerate(self.selected_users):
            user.b_t = 1
            user.tau = user.get_communication_round_time()
            user.r = 1 - user.tau / self.tau_max  # current user reward
            user.y_t = (user.y_t * user.z_t + user.r) / (user.z_t + 1)
            user.z_t += 1

            if user.tau >= self.tau_max:
                fails += 1

        self.fails.append(self.fails[-1] + fails)

    def save_results(self):
        file_name = dirname(__file__) + "/results/" + self.policy
        if self.beta:
            file_name += '_beta' + str(self.beta)
        file_name += ".h5"
        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset('sigma_pi', data=self.sigma_pi[1:])
            hf.create_dataset('fails', data=self.fails[1:])
            hf.create_dataset('selection_fraction', data=[user.z_t / self.num_glob_iters for user in self.users])


class User(object):
    """Base module for user"""

    def __init__(self, user_id, tau_max, alpha, c):
        np.random.seed(user_id)

        self.user_id = user_id
        self.batch_size = 6
        self.distance = np.random.rand()/2  # distance from AP in km
        self.tau_max = tau_max
        self.tau_d = None  # average downlink transmitting time in sec
        self.tau_u = None  # average uplink transmitting time in sec
        self.tau = None  # update time in sec
        self.mu = None  # real reward
        self.b_t = 0  # Indicator if user was selected in time t - 1
        self.y_t = 0  # UCB estimate reward in iteration t
        self.z_t = 0  # number of times that the user was selected
        self.y_hat_t = 1  # truncated UCB estimate reward in iteration t
        self.r = None  # current user reward
        if c:
            self.c = c[user_id]  # fairness restriction
        self.virtual_queue = 0
        self.alpha = alpha  # availability probability (binomal distribution)

        self.transmission_times()
        self.transmission_reward()

    def transmission_time(self):
        fading = 10 ** -12.81 * self.distance ** -3.76  # large scaling fading in dB
        gain = 23  # uplink and downlink transmitting power in dBm
        noise_power = -107  # std of noise in dBm
        bandwidth = 15e3  # allocated bandwidth in Hz
        message_size = 5e3  # size of transmitting model parameters
        average_rate = log2(1 + 10 ** (gain/10) * fading / 10 ** (noise_power/10))
        self.tau_d = message_size / (bandwidth * average_rate)
        self.tau_u = self.tau_d

    def transmission_reward(self):
        avg_tau_lu = self.batch_size / ((self.user_id + 2) * 10)  # average computing time in sec
        avg_tau = min(self.tau_d + self.tau_d + avg_tau_lu, self.tau_max)
        self.mu = 1 - avg_tau / self.tau_max

    def transmission_round_time(self):
        k = self.user_id
        tau_lu = self.batch_size / np.random.uniform(10*k + 10, 10*k + 30)  # realization of computing time in sec
        return min(self.tau_d + self.tau_d + tau_lu, self.tau_max)

    def check_availability(self):
        return np.random.binomial(1, self.alpha, 1)[0]

    def update_queue(self):
        self.virtual_queue = max(self.virtual_queue + self.c - self.b_t, 0)
