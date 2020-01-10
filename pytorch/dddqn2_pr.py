import argparse
import gym
import random
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import prioritized_replay as prioritized_replay

learning_rate = 0.0001
N_episode = 10000
max_epsilon = 0.08
min_epsilon = 0.001
T = 600
train_threshold = 2000
update_epoch = 10
copy_time = 50
SAVE_PATH = 'model/dddqn2_pr.pt'
batch_size = 64
buffer_limit = 65536
gamma = 0.99

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        ad_out = out[:-1]
        ad_out = ad_out - ad_out.mean()
        v_out = out[-1]
        q_out = ad_out + v_out
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return q_out.argmax().item()

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        return np.array(mini_batch)

    def size(self):
        return len(self.buffer)

def update(q_net, q_target_net, memory, optimizer, pr=False):
    for i in range(update_epoch):
        if pr:
            tree_idx, ISWeights, batch = memory.sample(batch_size)
        else:
            batch = memory.sample(batch_size)
        s, a, r, s_prime, done_mask = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]
        s = torch.tensor(s.tolist(), dtype=torch.float32)
        a = torch.tensor(a.tolist()).unsqueeze(-1)
        r = torch.tensor(r.tolist(), dtype=torch.float32).unsqueeze(-1)
        s_prime = torch.tensor(s_prime.tolist(), dtype=torch.float32)
        done_mask = torch.tensor(done_mask.tolist()).unsqueeze(-1)

        out = q_net(s)
        ad_out = out[:, :-1]
        ad_out = ad_out - ad_out.mean(1,keepdim=True)
        v_out = out[:, -1]
        v_out = torch.stack([v_out, v_out], dim=1)
        q_out = ad_out + v_out
        
        out_prime = q_net(s_prime)
        ad_out_prime = out_prime[:, :-1]
        ad_out_prime = ad_out_prime - ad_out_prime.mean(1,keepdim=True)
        v_out_prime = ad_out_prime[:, -1]
        v_out_prime = torch.stack([v_out_prime, v_out_prime], dim=1)
        q_out_prime = ad_out_prime + v_out_prime       

        target_out = q_target_net(s_prime)
        ad_target_out = target_out[:, :-1]
        ad_target_out = ad_target_out - ad_target_out.mean(1,keepdim=True)
        v_target_out = target_out[:, -1]
        v_target_out = torch.stack([v_target_out, v_target_out], dim=1)
        q_target_out = ad_target_out + v_target_out

        q_a = q_out.gather(1,a)
        max_a = q_out_prime.max(1)[1].unsqueeze(1)
        q_prime = q_target_out.gather(1, max_a)
        target = r + gamma * q_prime * done_mask

        if pr:
            abs_errors = torch.abs(target-q_prime).detach().numpy()
            memory.update_batch(tree_idx, abs_errors)
            loss = F.smooth_l1_loss(q_a, target, reduction='None')
            loss = torch.sum(loss*torch.tensor(ISWeights, dtype=torch.float))
        else:
            loss = F.smooth_l1_loss(q_a, target)
            # loss = F.mse_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(q_net.parameters(), 0.01)
        optimizer.step()

def train(pr=False):
    f = open('score/dddqn_pr_score.txt', 'a')
    f.seek(0)
    f.truncate()
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    if pr:
        memory = prioritized_replay.ReplayBuffer(buffer_limit)
    else:
        memory = ReplayBuffer(buffer_limit)
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
            memory.put((s.tolist(),a,r,s_prime.tolist(),done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() >= train_threshold:
            update(q, q_target, memory, optimizer, pr)

        if n_epi%copy_time == 0:
            q_target.load_state_dict(q.state_dict())

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
                  n_epi, score/print_interval, memory.size(), epsilon*100))
            f.write(str(score/print_interval) + ' ')
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
        score = 0.0
        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), min_epsilon)
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
    parser.add_argument('--pr', action='store_true', default=False, help='prioritized replay')
    args = parser.parse_args()

    if args.train:
        train(args.pr)
    else:
        evaluate()


if __name__ == '__main__':
    main()

'''
在使用优先经验回报时，经验回报池的大小一定要是2的次方的数，不然无法构成完全二叉数，也就没有办法构成
sum-tree...没办法构成完全二叉树绝对会报错！
'''