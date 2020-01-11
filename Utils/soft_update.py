import torch.nn as nn


def soft_update(net: nn.Module, net_target: nn.Module, tau: float = 0.005):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)