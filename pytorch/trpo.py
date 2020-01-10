import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distribution.Normal as Normal
from torch.autograd import Variable
from utils import *
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

class PI_net(nn.Module):
    def __init__(self):
        super(PI_net, self).__init__()
        self.fc_1 = nn.Linear(3, 128)
        self.fc_2 = nn.Linear(128, 32)
        self.fc_mu = nn.Linear(32, 1)
        self.fc_sigma = nn.Linear(32, 1)
    
    def forward(self, s):
        s = s.to(device)
        s = F.relu(self.fc_1(s))
        s = F.relu(self.fc_2(s))
        mu = torch.tanh(self.fc_mu(s))
        sigma = F.softplus(self.fc_sigma(s))
        pi_dist = Normal(mu, sigma)
        return pi_dist

class V_net(nn.Module):
    def __init__(self):
        super(V_net, self).__init__()
        self.fc_1 = nn.Linear(3, 128)
        self.fc_2 = nn.Linear(128, 32)
        self.fc_v = nn.Linear(32, 1)
    
    def forward(self, s)
        s = s.to(device)
        s = F.relu(self.fc_1(s))
        s = F.relu(self.fc_2(s))
        v = self.fc_mu(s)
        return v

class TRPO():
    def __init__(self):
        pi_net = PI_net()
        old_pi_net = PI_net()
        V_net = V_net()

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

    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    
    def get_loss(self, s, advantage):
        pi_log_prob = pi_net(s)
        old_pi_log_prob = old_pi_net(s)
        ratio = torch.exp(pi_dist.log_prob(a) - old_pi_dist.log_prob(a).detach())
        aloss = -ratio * advantage
        return aloss
    
    def Fvp(self, v):
            kl = get_kl()
            kl = kl.mean()

            grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, model.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + v * damping

    def conjugate_gradients(self, Avp, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = Avp(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
    return x

    def update(self):
        self.old_pi.load_state_dict(self.pi.state_dict())

        s, a, disc_r, s_prime, done_mask = self.make_batch()
        td_target = disc_r
        # TD0
        advantage = td_target - self.vn(s)
        advantage = advantage.detach()

        aloss = get_loss(s, advantage)
        aloss = aloss.to(device)
        aloss_grad = torch.autograd.grad(aloss, pi_net.parameters())

        # 利用共轭梯度算法计算梯度方向
        stepdir = conjugate_gradients(-aloss_grad, 10)

        shs = (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
        max_kl = 0
        # 拉格朗日乘子
        lm = 2 * torch.sqrt(max_kl / shs)

        fullstep = stepdir / lm

        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

        prev_params = get_flat_params_from(model)
        success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                        neggdotstepdir / lm[0])
        set_flat_params_to(model, new_params)


def train():
    env = gym.make('Pendulum-v0')
    model = TRPO()
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