import argparse
import time

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import sys
sys.path.append("..")

from Utils.normalization import NormFilter
from Utils.memory import Memory
from Utils.conjugate_gradient import conjugate_gradient, fvp
from Utils.soft_update import soft_update
from Utils.flatten import get_flat_params_from, set_flat_params_to
from Utils.distribution import normal_log_distribution

import pickle

# device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

SAVE_PATH = 'model/trpo.pt'
INPUT_DIM = 3
OUTPUT_DIM = 1
EPISODE_LENGTH = 500
EPISODE_NUM = 6000
GAMMA = 0.9
# TAU = 0.9
VALUE_LR = 0.0001     # 0.01
FVP_DAMPING = 0.01    # 0.0005
SOFT_UPDATE_TAU = 1   # doesn't work
MAX_KL = 0.01         # the smaller is better or 0.1
FILE_PATH = 'episode-reward'
NORM_FILTER_PICKLE_PATH = 'NormFilter'
torch.manual_seed(2020)


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.fc_1 = nn.Linear(INPUT_DIM, 128)
        self.fc_2 = nn.Linear(128, 32)
        self.mu = nn.Linear(32, OUTPUT_DIM)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        self.log_std = nn.Parameter(torch.zeros(1, OUTPUT_DIM, dtype=torch.float))

    def forward(self, s):
        s = torch.tanh(self.fc_1(s))
        s = torch.tanh(self.fc_2(s))
        mu = self.mu(s)
        log_std = 2 * torch.sigmoid(self.log_std).expand_as(mu)
        # log_std = self.log_std.expand_as(mu)
        std = torch.exp(log_std)
        return mu, log_std, std


class Value(nn.Module):

    def __init__(self):
        super(Value, self).__init__()
        self.fc_1 = nn.Linear(INPUT_DIM, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 32)
        self.value = nn.Linear(32, OUTPUT_DIM)
        self.value.weight.data.mul_(0.1)
        # self.value.bias.data.mul_(0.0)
    
    def forward(self, s):
        s = torch.relu(self.fc_1(s))
        s = torch.sigmoid(self.fc_2(s))
        s = torch.relu(self.fc_3(s))
        state_value = self.value(s)
        return state_value


class TRPO:
    def __init__(self):
        self.pi = Policy()
        self.old_pi = Policy()
        self.value = Value()
        self.v_optimizer = optim.Adam(self.value.parameters(), lr=VALUE_LR)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action_mean, _, action_std = self.pi(state)
        action = torch.normal(action_mean, action_std)
        return action

    def get_action_loss(self, states, actions, advantages):
        pi_mu, pi_logstd, pi_std = self.pi(states)
        log_pi_distri = normal_log_distribution(actions, pi_mu, pi_logstd, pi_std)
        old_pi_mu, old_pi_logstd, old_pi_std = self.old_pi(states)
        log_old_pi_distri = normal_log_distribution(actions, old_pi_mu, old_pi_logstd, old_pi_std)

        ratio = torch.exp(log_pi_distri - log_old_pi_distri)
        if torch.any(torch.isinf(ratio)):
            raise ValueError("ratio is inf")
        action_loss = -ratio * advantages.detach()
        return action_loss.mean()

    def get_kl(self, state):
        mean, log_std, std = self.pi(state)

        mean0, log_std0, std0 = self.old_pi(state)
        mean0 = mean0.detach()
        log_std0 = log_std0.detach()
        std0 = std0.detach()

        kl = log_std0 - log_std + (std.pow(2) + (mean - mean0).pow(2)) / (2.0 * std0.pow(2)) - 0.5

        return kl.sum(1, keepdim=True)

    def linesearch(self, states, actions, advantages, params, fullstep, expected_improve_rate,
                   max_backtracks=10, accept_ratio=.02):
        fval = self.get_action_loss(states, actions, advantages)
        for (_n_backtracks, step_frac) in enumerate(.5 ** np.arange(max_backtracks)):
            new_params = params + step_frac * fullstep
            set_flat_params_to(self.pi, new_params)
            newfval = self.get_action_loss(states, actions, advantages)
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * step_frac
            ratio = actual_improve / expected_improve
            # print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                return True, new_params
        return False, params

    def update(self, batch):
        states, actions, rewards, states_prime, done_masks = batch
        values = self.value(states)
        values_prime = self.value(states_prime)
        returns = torch.empty((rewards.size(0), 1), dtype=torch.float)
        # deltas = torch.empty((rewards.size(0), 1), dtype=torch.float)
        advantages = torch.empty((rewards.size(0), 1), dtype=torch.float)
        prev_return = 0.0
        # prev_advantage = 0.0

        # TD0
        for i in reversed(range(rewards.size(0))):
            # GAE(doesn't work)
            returns[i] = rewards[i] + GAMMA * prev_return * done_masks[i].item()
            # deltas[i] = rewards[i] + GAMMA * prev_value * done_masks[i].item() - values[i]
            # advantages[i] = deltas[i] + GAMMA * TAU * prev_advantage * done_masks[i].item()

            # returns[i] = rewards[i] + GAMMA * values_prime[i]
            advantages[i] = rewards[i] + GAMMA * values_prime[i] * done_masks[i].item() - values[i]

            prev_return = returns[i][0]
            # prev_advantage = advantages[i][0]

        # update value net
        targets = returns
        v_loss = F.smooth_l1_loss(values, targets.detach())
        self.v_optimizer.zero_grad()
        v_loss.mean().backward()
        for _ in range(10):
            self.v_optimizer.step()
        # print('v_loss = ', v_loss.mean())

        # update policy net using TRPO
        a_loss = self.get_action_loss(states, actions, advantages)
        if torch.any(torch.isnan(a_loss)):
            raise ValueError("a_loss is nan")
        a_loss_grad = torch.autograd.grad(a_loss, self.pi.parameters())
        a_loss_grad_flat = torch.cat([grad.view(-1) for grad in a_loss_grad])
        if torch.any(torch.isnan(a_loss_grad_flat)):
            raise ValueError("a_loss_grad_flat is nan")

        # 利用共轭梯度算法计算梯度方向
        step_dir = conjugate_gradient(model=self.pi, kl=self.get_kl(states), b=-a_loss_grad_flat,
                                      nsteps=10, fvp_damping=FVP_DAMPING)

        shs = 0.5 * (step_dir * fvp(self.get_kl(states), self.pi, step_dir)).sum(0, keepdim=True)
        if np.sign(shs.item()) == -1:
            lm = torch.tensor(np.inf, dtype=float)
        # 拉格朗日乘子
        else:
            lm = torch.sqrt(2 * MAX_KL / shs)
        full_step = step_dir / lm  # lm
        neg_g_dot_step_dir = (-a_loss_grad_flat * step_dir).sum(0, keepdim=True)

        prev_params = get_flat_params_from(self.pi)
        success, new_params = self.linesearch(states, actions, advantages, prev_params,
                                              full_step, neg_g_dot_step_dir / lm)
        if not success:
            print('linesearch fail!')
        set_flat_params_to(self.pi, new_params)


def train():
    env = gym.make('Pendulum-v0')
    env.seed(2020)
    fp = open(FILE_PATH, 'w')
    pickle_file = open(NORM_FILTER_PICKLE_PATH, 'wb')
    norm_filter_state = NormFilter((INPUT_DIM,), clip=5)
    # norm_filter_reward = NormFilter((1,), decorate_mean=False, clip=10)
    trpo_model = TRPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(EPISODE_NUM):
        s = env.reset()
        s = norm_filter_state(s)
        memory = Memory()
        for _ in range(EPISODE_LENGTH):
            a = trpo_model.choose_action(s)
            a = a.detach()[0].numpy()
            s_prime, r, done, info = env.step(a)
            score += r

            r = np.array([r], dtype=np.float)
            done = np.array([done])
            s_prime = norm_filter_state(s_prime.squeeze())
            memory.put((s, a, r/100.0, s_prime, 1 - done))
            s = s_prime

            if done:
                break

        batch = memory.sample_all()
        # print(batch)
        trpo_model.update(batch)
        soft_update(trpo_model.pi, trpo_model.old_pi, SOFT_UPDATE_TAU)
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            fp.write("{}, {}\n".format(n_epi, score/print_interval))
            score = 0.0

    torch.save({
        'trpo_model_pi_dict': trpo_model.pi.state_dict(),
        'trpo_model_value_dict': trpo_model.value.state_dict(),
        }, SAVE_PATH)
    pickle.dump(norm_filter_state, pickle_file)

    env.close()
    fp.close()
    pickle_file.close()


def evaluate():
    env = gym.make('Pendulum-v0')
    env.seed(2020)
    checkpoint = torch.load(SAVE_PATH)
    pi = Policy()
    pi.load_state_dict(checkpoint['trpo_model_pi_dict'])
    pi.eval()
    pickle_file = open(NORM_FILTER_PICKLE_PATH, 'rb')
    norm_filter_state = pickle.load(pickle_file)

    start = time.time()
    end = time.time()
    while end - start < 60:
        s = env.reset()
        s = norm_filter_state(s, update=False)
        done = False
        reward_sum = 0
        while not done:
            s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
            action_mean, _, action_std = pi(s)
            a = torch.normal(action_mean, action_std)
            a = a.detach()[0].numpy()
            s_prime, r, done, info = env.step(a)
            s_prime = norm_filter_state(s_prime, update=False)
            env.render()
            s = s_prime
            reward_sum += r
        print('done! reward = ', reward_sum)
        end = time.time()

    env.close()
    pickle_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='training process')
    args = parser.parse_args()
    if args.train:
        train()
    else:
        evaluate()


if __name__ == '__main__':
    main()


'''
CONCLUTION:
    1. GAE doesn't work, because it has large variance, I think.
    2. The MAX_KL constraint condition should be set as smaller as you can.
    3. running_state algorithm (also is seen as normalization filter) may be helpful.
    4. It's not promised that the deeper network is the better.
'''