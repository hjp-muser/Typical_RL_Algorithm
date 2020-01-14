import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np

#Hyperparameters
pi_lr = 0.0001
vn_lr = 0.0002
gamma = 0.9
lmbda = 0.8
eps_clip = 0.05
pi_epoch = 10
vn_epoch = 10
batch_size = 32
N = 5000
SAVE_PATH = 'model/ppo.pt'

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc_1 = nn.Linear(3, 128)
        self.fc_2 = nn.Linear(128, 32)
        self.fc_mu = nn.Linear(32, 1)
        self.fc_sigma = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc_1(x))
        x = torch.relu(self.fc_2(x))
        mu = torch.tanh(self.fc_mu(x))
        sigma = torch.softplus(self.fc_sigma(x))
        pi_dist = torch.distributions.Normal(mu, sigma)
        return pi_dist

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.fc_1 = nn.Linear(3, 128)
        self.fc_2 = nn.Linear(128, 32)
        # self.fc_3 = nn.Linear(64, 32)
        self.fc_v = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc_1(x))
        x = torch.relu(self.fc_2(x))
        v = self.fc_v(x)
        return v

class PPO():
    def __init__(self):
        self.data = []
        self.pi = Policy()
        self.old_pi = Policy()
        self.vn = Value()
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=pi_lr)
        self.vn_optimizer = optim.Adam(self.vn.parameters(), lr=vn_lr)
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        
        disc_r_lst = []
        r_lst = np.array(r_lst)
        # 利用 V 网络对当前状态的值进行估计，求折扣回报
        disc_r = self.vn(torch.tensor([s_prime_lst[-1]], dtype=torch.float)) 
        for r in r_lst[::-1]:
            disc_r = r[0] + gamma * disc_r  # * done_lst[-1][0]
            disc_r_lst.append([disc_r])
        disc_r_lst.reverse()

        s,a,disc_r,s_prime,done_mask = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(disc_r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s, a, disc_r, s_prime, done_mask
        
    def update(self):
        self.old_pi.load_state_dict(self.pi.state_dict())

        s, a, disc_r, s_prime, done_mask = self.make_batch()
        td_target = disc_r
        # TD0
        advantage = td_target - self.vn(s)
        advantage = advantage.detach()
        # GAE
        # delta = td_target - self.vn(s)
        # delta = delta.detach().numpy()
        # advantage_lst = []
        # advantage = 0.0
        # for delta_t in delta[::-1]: # delta 逆序
        #     advantage = gamma * lmbda * advantage + delta_t[0]
        #     advantage_lst.append([advantage])
        # advantage_lst.reverse()
        # advantage = torch.tensor(advantage_lst, dtype=torch.float)
        # 更新策略网络
        for _ in range(pi_epoch):
            pi_dist = self.pi(s)
            old_pi_dist = self.old_pi(s)
            ratio = torch.exp(pi_dist.log_prob(a) - old_pi_dist.log_prob(a))  # a/b == exp(log(a)-log(b))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            aloss = -torch.min(surr1, surr2)
            self.pi_optimizer.zero_grad()
            aloss.mean().backward()
            self.pi_optimizer.step()
        # 更新价值网络
        for _ in range(vn_epoch):
            closs = F.smooth_l1_loss(self.vn(s), td_target.detach())
            # closs = F.smooth_l1_loss(self.vn(s) , td_target.detach())
            self.vn_optimizer.zero_grad()
            closs.mean().backward()
            self.vn_optimizer.step()
    
    def choose_action(self, x):
        pi_dist = self.pi(torch.from_numpy(x).float())
        a = pi_dist.sample()
        return np.clip(a, -2, 2) # 这里注意裁剪
        
def train():
    env = gym.make('Pendulum-v0')
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(N):
        s = env.reset()
        done = False
        # for t in range(200):
        while not done:
            for _ in range(batch_size):
                a= model.choose_action(s)
                s_prime, r, done, info = env.step(a)
                model.put_data((s, a, (r+8)/8, s_prime, done))
                s = s_prime

                score += r

                if done:
                    break
            # if (t+1) % batch_size == 0 or t == 200-1:
            model.update()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    torch.save({
        'ppo_model_dict': model.state_dict(),
        }, SAVE_PATH)

    env.close()


def evaluate():
    env = gym.make('Pendulum-v0')
    checkpoint = torch.load(SAVE_PATH)
    model = PPO()
    model.load_state_dict(checkpoint['ppo_model_dict'])
    model.eval()
    while 1:
        s = env.reset()
        done = False
        reward_sum = 0
        while not done:
            _, a = model.pi(torch.from_numpy(s).float()) 
            s_prime, r, done, info = env.step([a.item()])
            env.render()
            s = s_prime
            reward_sum += r
        print('done! reward = ', reward_sum)
        
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
''' 