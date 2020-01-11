import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update
update_epoch = 10
train_threshold = 2000
T = 300
N = 5000
SAVE_PATH = 'model/ddpg.pt'

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)
        #initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))*2 # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        
        # self.fc_s = nn.Linear(3, 64)
        # self.fc_a = nn.Linear(1,64)
        self.fc_sa = nn.Linear(4, 128)
        self.fc_q = nn.Linear(128, 64) #32
        self.fc_3 = nn.Linear(64,1)
        #initialization

    def forward(self, x, a):
        # h1 = F.relu(self.fc_s(x))
        # h2 = F.relu(self.fc_a(a))
        # cat = torch.cat([h1,h2], dim=1)
        cat = torch.cat([x, a], dim=1)
        q = F.relu(self.fc_sa(cat))
        
        q = F.relu(self.fc_q(q))
        q = self.fc_3(q)
        return q


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def update(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask  = memory.sample(batch_size)
    
    target = r + gamma * q_target(s_prime, mu_target(s_prime))
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s, mu(s)).mean()  # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def train():
    env = gym.make('Pendulum-v0')
    memory = ReplayBuffer()

    q, q_target = QNet(), QNet()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(N):
        s = env.reset()
        
        for t in range(T): # maximum length of episode is 200 for Pendulum-v0
            a = mu(torch.from_numpy(s).float()) 
            a = a.item() + ou_noise()[0]
            s_prime, r, done, info = env.step([a])
            memory.put((s,a,r/100.0,s_prime,done))
            score +=r
            s = s_prime

            if done:
                break              
                
        if memory.size() > train_threshold:
            for i in range(update_epoch):
                update(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    torch.save({
            'q_state_dict': q.state_dict(),
            'q_target_state_dict': q_target.state_dict(),
            'mu_state_dict': mu.state_dict(),
            'mu_target_state_dict': mu_target.state_dict(),
            'q_optimizer_state_dict': q_optimizer.state_dict(),
            'mu_optimizer_state_dict': mu_optimizer.state_dict()
            }, SAVE_PATH)


def evaluate():
    env = gym.make('Pendulum-v0')
    checkpoint = torch.load(SAVE_PATH)
    mu = MuNet()
    mu.load_state_dict(checkpoint['mu_state_dict'])
    mu.eval()
    while 1:
        s = env.reset()
        done = False
        reward_sum = 0
        while not done:
            a = mu(torch.from_numpy(s).float())
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