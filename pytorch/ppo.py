import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

#Hyperparameters
learning_rate = 0.0001
gamma         = 0.9
lmbda         = 0.8
eps_clip      = 0.1
K_epoch       = 10
T_horizon     = 32
N = 5000
SAVE_PATH = 'model/ppo.pt'

class PPO(nn.Module):
    def __init__(self,device):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc_1 = nn.Linear(3,128)
        self.fc_2 = nn.Linear(128, 32) 
        # self.fc_3 = nn.Linear(256, 64)
        # self.fc_4 = nn.Linear(64, 16)
        self.fc_mu = nn.Linear(32, 1)
        self.fc_v = nn.Linear(32, 1)
        self.var = 1
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = device

    def pi(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        mu = self.fc_mu(x)
        prob = torch.distributions.Normal(mu, self.var)
        sample = prob.sample()
        log_prob = prob.log_prob(sample)
        a = torch.tanh(sample) * 2
        return log_prob, a

    def v(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, log_prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, log_prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            log_prob_a_lst.append([log_prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, log_prob_a = torch.tensor(s_lst, dtype=torch.float).to(self.device), \
                                          torch.tensor(a_lst).to(self.device), \
                                          torch.tensor(r_lst).to(self.device), \
                                          torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
                                          torch.tensor(done_lst, dtype=torch.float).to(self.device), \
                                          torch.tensor(log_prob_a_lst).to(self.device)
        self.data = []
        return s, a, r, s_prime, done_mask, log_prob_a
        
    def update(self):
        s, a, r, s_prime, done_mask, log_prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) # * done_mask

            delta = td_target - self.v(s)
            delta = delta.detach().cpu().numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]: # delta 逆序
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
            pi_log_prob, _ = self.pi(s)
            ratio = torch.exp(pi_log_prob - log_prob_a)  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
def train(device):
    env = gym.make('Pendulum-v0')
    model = PPO(device)
    model.to(device)
    score = 0.0
    print_interval = 20

    for n_epi in range(N):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                log_prob, a = model.pi(torch.from_numpy(s).float())
                log_prob = log_prob.cpu()
                a = a.cpu()
                s_prime, r, done, info = env.step(a)
                model.put_data((s, a, r, s_prime, log_prob, done))
                s = s_prime

                score += r
                if done:
                    break

            model.update()
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

#     torch.save({
#         'ppo_model_dict': model.state_dict(),
#         }, SAVE_PATH)

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
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    if args.train:
        train(device)
    else:
        evaluate()


if __name__ == '__main__':
    main()