import argparse
import time

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from Utils.normalization import NormFilter
from Utils.memory import Memory
from Utils.conjugate_gradient import conjugate_gradient, fvp
from Utils.soft_update import soft_update

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


SAVE_PATH = 'model/trpo.pt'
INPUT_DIM = 3
OUTPUT_DIM = 1
EPISODE_LENGTH = 300
EPISODE_NUM = 5000
BATCH_SIZE = 32
GAMMA = 0.995
TAU = 0.9
VALUE_LR = 0.0001
FVP_DAMPING = 0.01
SOFT_UPDATE_TAU = 0.005


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.fc_1 = nn.Linear(INPUT_DIM, 128)
        self.fc_2 = nn.Linear(128, 32)
        self.mu = nn.Linear(32, OUTPUT_DIM)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        self.log_std = nn.Parameter(torch.zeros(1, OUTPUT_DIM))

    def forward(self, s):
        # s = s.to(device)
        s = F.relu(self.fc_1(s))
        s = F.relu(self.fc_2(s))
        # mu = torch.tanh(self.mu(s))
        mu = self.mu(s)
        log_std = self.log_std.expand_as(mu)
        std = torch.exp(log_std)
        return mu, log_std, std


class Value(nn.Module):

    def __init__(self):
        super(Value, self).__init__()
        self.fc_1 = nn.Linear(INPUT_DIM, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.value = nn.Linear(64, OUTPUT_DIM)
        self.value.weight.data.mul_(0.1)
        self.value.weight.data.mul_(0.0)
    
    def forward(self, s):
        # s = s.to(device)
        s = F.relu(self.fc_1(s))
        s = F.relu(self.fc_2(s))
        state_value = self.value(s)
        return state_value


class TRPO:
    def __init__(self):
        self.pi = Policy()
        self.old_pi = Policy()
        self.value = Value()
        self.v_optimizer = optim.Adam(self.value.parameters(), lr=VALUE_LR)

    def choose_action(self, state):
        action_mean, _, action_std = self.pi(state)
        action = torch.normal(action_mean, action_std)
        return action

    def get_action_loss(self, states, advantages):
        pi_mu, pi_std, pi_logstd = self.pi(states)
        old_pi_mu, old_pi_std, old_pi_logstd = self.old_pi(states)
        ratio = torch.exp(pi_logstd - old_pi_logstd.detach())
        action_loss = -ratio * advantages
        return action_loss.mean()

    def get_kl(self, state):
        mean1, log_std1, std1 = self.pi(state)

        # mean0 = Variable(mean1.data)
        mean0 = torch.tensor(mean1)
        log_std0 = torch.tensor(log_std1)
        # std0 = Variable(std1.data)
        std0 = torch.tensor(std1)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def update(self, batch):
        states, actions, rewards, state_primes, done_masks = batch

        values = self.value(states)
        returns = torch.empty(rewards.size(0), 1)
        deltas = torch.empty(rewards.size(0), 1)
        advantages = torch.empty(rewards.size(0), 1)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        # TD0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + GAMMA * prev_return * done_masks[i]
            deltas[i] = rewards[i] + GAMMA * prev_value * done_masks[i] - values[i]
            advantages[i] = deltas[i] + GAMMA * TAU * prev_advantage * done_masks[i]

            prev_return = returns[i][0]
            prev_value = values[i][0]
            prev_advantage = advantages[i][0]

        # update value net
        targets = returns
        v_loss = F.smooth_l1_loss(values, targets.detach())
        self.v_optimizer.zero_grad()
        v_loss.mean().backward()
        self.v_optimizer.step()

        # update policy net using TRPO
        a_loss = self.get_action_loss(states, advantages)
        a_loss.mean().backward()
        a_loss_grad = a_loss.grad

        # 利用共轭梯度算法计算梯度方向
        step_dir = conjugate_gradient(self.get_kl(), -a_loss_grad, 10, FVP_DAMPING)

        shs = (step_dir * fvp(self.get_kl(), step_dir)).sum(0, keepdim=True)
        max_kl = 0
        # 拉格朗日乘子
        lm = 2 * torch.sqrt(max_kl / shs)

        fullstep = step_dir / lm

        neggdotstepdir = (-loss_grad * step_dir).sum(0, keepdim=True)
        print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

        prev_params = get_flat_params_from(model)
        success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                        neggdotstepdir / lm[0])
        set_flat_params_to(model, new_params)


def train():
    env = gym.make('Pendulum-v0')
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
            a = a.float()
            s_prime, r, done, info = env.step(a)
            s_prime = norm_filter_state(s_prime)
            memory.put((s, a, (r+8)/8, s_prime, 1 - done))
            s = s_prime

            score += r

            if done:
                break

        batch = memory.sample_all()
        trpo_model.update(batch)
        soft_update(trpo_model.old_pi, trpo_model.pi, SOFT_UPDATE_TAU)
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    torch.save({
        'trpo_model_pi_dict': trpo_model.pi.state_dict(),
        'trpo_model_value_dict': trpo_model.value.state_dict(),
        }, SAVE_PATH)

    env.close()


def evaluate():
    env = gym.make('Pendulum-v0')
    checkpoint = torch.load(SAVE_PATH)
    pi = Policy()
    pi.load_state_dict(checkpoint['pi_state_dict'])
    pi.eval()
    start = time.time()
    end = time.time()
    while end - start < 60:
        s = env.reset()
        done = False
        reward_sum = 0
        while not done:
            a = pi(torch.from_numpy(s).float())
            s_prime, r, done, info = env.step([a.item()])
            env.render()
            s = s_prime
            reward_sum += r
        print('done! reward = ', reward_sum)
        end = time.time()

    env.close()


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