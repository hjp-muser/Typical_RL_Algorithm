import torch
import torch.nn as nn


def fvp(kl: torch.Tensor, model: nn.Module, v: torch.Tensor, damping: float = 0.01) -> float:
    """
    fisher-vector product
    :param model:
    :param kl:
    :param v:
    :param damping: That is just a damping technique used in conjugate gradient methods to make it more stable.
    :return:
    """
    kl = kl.mean()
    kl_grads = torch.autograd.grad(kl, model.parameters(), create_graph=True, retain_graph=True)
    kl_grads_flat = torch.cat([grad.view(-1) for grad in kl_grads])
    kl_v = kl_grads_flat * v
    kl_v = kl_v.sum()
    kl_v_grads = torch.autograd.grad(kl_v, model.parameters(), retain_graph=True)
    kl_v_grads_flat = torch.cat([grad.view(-1) for grad in kl_v_grads])
    return kl_v_grads_flat + v * damping


def conjugate_gradient(model, kl, b, nsteps, fvp_damping=0.01, residual_tol=1e-10):
    """
    theory: https://blog.csdn.net/lusongno1/article/details/78550803
    :param model:
    :param kl:
    :param b:
    :param nsteps:
    :param fvp_damping:
    :param residual_tol:
    :return:
    """
    if torch.any(torch.isnan(b)):
        raise ValueError("b is nan")
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    r_dot_r = torch.dot(r, r)
    for i in range(nsteps):
        _fvp = fvp(kl, model, p, fvp_damping)
        alpha = r_dot_r / torch.dot(p, _fvp)
        x += alpha * p
        r -= alpha * _fvp
        new_r_dot_r = torch.dot(r, r)
        betta = new_r_dot_r / r_dot_r
        p = r + betta * p
        r_dot_r = new_r_dot_r
        if r_dot_r < residual_tol:
            break
    return x