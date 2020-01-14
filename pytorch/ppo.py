import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import time
import sys

sys.path.append("..")

from Utils.distribution import normal_log_distribution
from Utils.memory import Memory

pi_lr = 0.0001
vn_lr = 0.0001
gamma = 0.9
eps_clip = 0.01
pi_epoch = 10
vn_epoch = 10
batch_size = 128
N = 5000
SAVE_PATH = 'model/ppo.pt'
torch.manual_seed(2020)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.mu = nn.Linear(32, 1)
        self.log_std = nn.Parameter(torch.zeros(1, 1))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        mu = self.mu(x)
        log_std = 2 * torch.sigmoid(self.log_std)
        std = torch.exp(log_std)
        return mu, log_std, std


class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.value = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.relu(self.fc3(x))
        v = self.value(x)
        return v


class PPO:
    def __init__(self):
        self.pi = Policy()
        self.old_pi = Policy()
        self.vn = Value()
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=pi_lr)
        self.vn_optimizer = optim.Adam(self.vn.parameters(), lr=vn_lr)
        self.memory = None

    def update(self):
        self.old_pi.load_state_dict(self.pi.state_dict())
        states, actions, rewards, states_prime, done_masks = self.memory.sample_all()
        return_lst = torch.zeros_like(rewards)
        prev_return = 0.0
        for i in reversed(range(rewards.size(0))):
            return_lst[i] = rewards[i] + gamma * prev_return * done_masks[i].item()
            prev_return = return_lst[i][0]
        target = return_lst
        # TD0
        advantage = rewards + gamma * self.vn(states_prime) * done_masks[i].item() - self.vn(states)
        advantage = advantage.detach()

        # 更新策略网络
        pi_mu, pi_log_std, pi_std = self.pi(states)
        old_pi_mu, old_pi_log_std, old_pi_std = self.old_pi(states)
        pi_log_distri = normal_log_distribution(actions, pi_mu, pi_log_std, pi_std)
        old_pi_log_distri = normal_log_distribution(actions, old_pi_mu, old_pi_log_std, old_pi_std)
        ratio = torch.exp(pi_log_distri - old_pi_log_distri)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
        aloss = -torch.min(surr1, surr2)
        for _ in range(pi_epoch):
            self.pi_optimizer.zero_grad()
            aloss.mean().backward(retain_graph=True)
            self.pi_optimizer.step()

        # 更新价值网络
        closs = F.smooth_l1_loss(self.vn(states), target.detach())
        for _ in range(vn_epoch):
            self.vn_optimizer.zero_grad()
            closs.mean().backward(retain_graph=True)
            self.vn_optimizer.step()

    def choose_action(self, x):
        x = x[np.newaxis, :]
        mu, _, std = self.pi(torch.from_numpy(x).float())
        a = torch.normal(mu, std)
        return a


def train():
    env = gym.make('Pendulum-v0')
    env.seed(2020)
    ppo_model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(N):
        s = env.reset()
        ppo_model.memory = Memory()
        # while not done:
        #     for _ in range(batch_size):
        for t in range(500):
            a = ppo_model.choose_action(s)
            a = a.detach()[0].numpy()
            s_prime, r, done, info = env.step(a)
            done_mask = 0 if done else 1
            ppo_model.memory.put((s, a, [(r + 8) / 8], s_prime, [done_mask]))
            s = s_prime
            score += r
            if done:
                break
            # if (t+1) % batch_size == 0 or t == 200-1:
            # ppo_model.update()
        ppo_model.update()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    torch.save({
        'ppo_policy_dict': ppo_model.pi.state_dict(),
    }, SAVE_PATH)

    env.close()


def evaluate():
    env = gym.make('Pendulum-v0')
    env.seed(2020)
    checkpoint = torch.load(SAVE_PATH)
    policy = Policy()
    policy.load_state_dict(checkpoint['ppo_policy_dict'])
    policy.eval()
    start_time = time.time()
    end_time = time.time()
    while end_time - start_time < 60:
        s = env.reset()
        done = False
        reward_sum = 0
        while not done:
            mu, _, std = policy(torch.from_numpy(s).float())
            s_prime, r, done, info = env.step(mu.detach().numpy())
            env.render()
            s = s_prime
            reward_sum += r
        print('done! reward = ', reward_sum)
        end_time = time.time()

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

'''
1. 计算折扣回报的方式：
    v_s_ = r + gamma * self.v(s_prime) * done_mask
这种方式是计算每个折扣回报，v 网络都有参与估计 s_prime 的值

更新计算折扣回报的方式：
    v_s_ = self.v(s_prime)
    v_s_ = r + gamma * v_s_ * done_mask
这种方式只利用 v 网络估计当前状态

第二种方式更好
-----------

2. 什么时候更新价值网络
- 更新一次策略网络就更新一次价值网络

- 更新完策略网络，再更新价值网络 

目前用第二种方式
----------------

3. 策略网络和价值网络，更新迭代的次数设置为多少：
- 设置相同，效果一般
- 策略网络更新的次数多，效果不好
- 价值网络更新的次数多，效果还行
-----------------

4. 动作值的范围忘了裁减了。。。。。这非常重要，裁剪到 [-2, 2]
还有 mu 经过 tanh 函数激活后也要乘以 2 。。

-----------------

5. 计算折扣回报的时候，结束状态需不需要乘以 donemask ?
不需要的效果会好很多。。。有可能只是特例而已
!!!!!! 这个影响很大？为什么？？？结束状态难道不需要乘以 donemask 吗

-----------------
6. gamma 的取值也会影响收敛：
如果不收敛，适当调小一点？

-----------------
7. geng xin bu neng zhi yong yi ge episode de bu fen shu ju
'''


