import argparse
import gym
import random
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

learning_rate = 0.0002
N_episode = 10000
max_epsilon = 0.08
min_epsilon = 0.001
T = 600
train_threshold = 2000
update_epoch = 10
copy_time = 50
SAVE_PATH = 'model/dddqn.pt'
batch_size = 64
buffer_limit = 50000
gamma = 0.99

class Anet(nn.Module):
    def __init__(self):
        super(Anet, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Vnet(nn.Module):
    def __init__(self):
        super(Vnet, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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

        return torch.tensor(s_lst, dtype=torch.float), \
                torch.tensor(a_lst), \
                torch.tensor(r_lst, dtype=torch.float), \
                torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


def sample_action(ad_net, v_net, obs, epsilon):
    ad_out = ad_net(obs)
    ad_out = ad_out - ad_out.mean()
    v_out = v_net(obs)
    q_out = ad_out + v_out
    coin = random.random()
    if coin < epsilon:
        return random.randint(0, 1)
    else:
        return q_out.argmax().item()


def update(ad_net, ad_target_net, v_net, v_target_net, memory, ad_optimizer, v_optimizer):
    for i in range(update_epoch):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        ad_out = ad_net(s)
        ad_out = ad_out - ad_out.mean(1,keepdim=True)
        v_out = v_net(s)
        q_out = ad_out + v_out

        ad_out_prime = ad_net(s_prime)
        ad_out_prime = ad_out_prime - ad_out_prime.mean(1,keepdim=True)
        v_out_prime = v_net(s_prime)
        q_out_prime = ad_out_prime + v_out_prime       

        ad_target_out = ad_target_net(s_prime)
        ad_target_out = ad_target_out - ad_target_out.mean(1,keepdim=True)
        v_target_out = v_target_net(s_prime)
        q_target_out = ad_target_out + v_target_out

        q_a = q_out.gather(1,a)
        max_a = q_out_prime.max(1)[1].unsqueeze(1)
        q_prime = q_target_out.gather(1, max_a)
        target = r + gamma * q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        ad_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(ad_net.parameters(), 0.01)
        ad_optimizer.step()
        v_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(v_net.parameters(), 0.01)
        v_optimizer.step()

def train():
    env = gym.make('CartPole-v1')
    ad = Anet()
    ad_target = Anet()
    v = Vnet()
    v_target = Vnet() 
    ad_target.load_state_dict(ad.state_dict())
    v_target.load_state_dict(v.state_dict())
    memory = ReplayBuffer()
    ad_optimizer = optim.Adam(ad.parameters(), lr=learning_rate)
    v_optimizer = optim.Adam(v.parameters(), lr=learning_rate)
    print_interval = 20
    score = 0.0

    for n_epi in range(N_episode):
        # epsilon = max(min_epsilon, max_epsilon - 0.01  * (n_epi / 200))
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon)  * (n_epi / N_episode))
        s = env.reset()

        for t in range(T):
            a = sample_action(ad, v, torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r,s_prime,done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > train_threshold:
            update(ad, ad_target, v, v_target, memory, ad_optimizer, v_optimizer)

        if n_epi%copy_time == 0:
            ad_target.load_state_dict(ad.state_dict())
            v_target.load_state_dict(v.state_dict())

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
                  n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0

    torch.save(
    {
        'ad_state_dict': ad.state_dict(),
        'ad_target_state_dict': ad_target.state_dict(),
        'v_state_dict': v.state_dict(),
        'v_target_state_dict': v_target.state_dict(),
        'ad_optim_state_dict': ad_optimizer.state_dict(),
        'v_optim_state_dict': v_optimizer.state_dict()
    }, SAVE_PATH)
    env.close()


def evaluate():
    ad = Anet()
    v = Vnet()
    checkpoint = torch.load(SAVE_PATH)
    ad.load_state_dict(checkpoint['ad_state_dict'])
    ad.eval()
    v.load_state_dict(checkpoint['v_state_dict'])
    v.eval()
    env = gym.make('CartPole-v1')
    while(1):
        s = env.reset()
        done = False
        score = 0.0
        while not done:
            a = sample_action(ad, v, torch.from_numpy(s).float(), min_epsilon)
            s_prime, r, done, info = env.step(a)
            score += r
            env.render()    
            s = s_prime
        print('score: ', score)
        print('done!')

    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    args = parser.parse_args()

    if args.train:
        train()
    else:
        evaluate()


if __name__ == '__main__':
    main()

'''
学习效果还没有DDQN的好，学习率调小到了0.0001，不过在迭代次数为5000的时候，居然学习出最好的分数，但是继续迭代效果反而变差。

后来学习率调成 0.0005->0.0002,batch_size大小调为32->64,最终的探索率调为 0.01->0.001，效果比DDQN要好。。

batch_size为32时会造成过拟合，分数会迅速彪得很高，但是不久就降了下来，而且还升不上去。

进行梯度裁剪后的效果更好，梯度上限设置为0.01。
'''