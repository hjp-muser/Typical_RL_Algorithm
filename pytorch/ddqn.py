import argparse
import gym
import random
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

learning_rate = 0.0002  #0.0005
N_episode = 10000
max_epsilon = 0.08
min_epsilon = 0.001   #0.01
T = 600
train_threshold = 2000
update_epoch = 10
copy_time = 50
SAVE_PATH = 'model/dqqn.pt'
batch_size = 64  #32
buffer_limit = 50000
gamma = 0.99

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


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

def update(q, q_target, memory, optimizer):
    for i in range(update_epoch):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_a  = q(s_prime).max(1)[1].unsqueeze(1)
        q_prime = q_target(s_prime).gather(1,max_a)
        target = r + gamma * q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train():
    f = open('score/ddqn_score.txt', 'a')
    f.seek(0)
    f.truncate()
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    print_interval = 20
    score = 0.0

    for n_epi in range(N_episode):
        # epsilon = max(min_epsilon, max_epsilon - 0.01  * (n_epi / 200))
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon)  * (n_epi / N_episode))
        s = env.reset()

        for t in range(T):
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r,s_prime,done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > train_threshold:
            update(q, q_target, memory, optimizer)

        if n_epi%copy_time == 0:
            q_target.load_state_dict(q.state_dict())

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
                  n_epi, score/print_interval, memory.size(), epsilon*100))
            f.write(str(score/print_interval)+' ')
            score = 0.0

    torch.save(
    {
        'q_state_dict': q.state_dict(),
        'q_target_state_dict': q_target.state_dict(),
        'optim_state_dict': optimizer.state_dict()
    }, SAVE_PATH)
    env.close()
    f.close()


def evaluate():
    q = Qnet()
    checkpoint = torch.load(SAVE_PATH)
    q.load_state_dict(checkpoint['q_state_dict'])
    q.eval()
    env = gym.make('CartPole-v1')
    while(1):
        s = env.reset()
        done = False
        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), min_epsilon)
            s_prime, r, done, info = env.step(a)
            env.render()    
            s = s_prime
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
PS:
改成DDQN后，使用与DQN相同的参数，效果还没有DQN的好。。。在此mark一下
不过调高了学习率以后，效果好多了
'''