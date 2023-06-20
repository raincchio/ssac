import torch
import utils.pytorch_util as ptu
from model.policies import TanhNormal
import math


def get_bothq_exploration_action(ob_np, policy=None, qfs=None, hyper_params=None):

    assert ob_np.ndim == 1

    beta_UB = 4.66
    delta = 23.53

    ob = ptu.from_numpy(ob_np)

    # Ensure that ob is not batched
    assert len(list(ob.shape)) == 1

    _, pre_tanh_mu_T, _, _, std, _ = policy(ob)

    # Ensure that pretanh_mu_T is not batched
    assert len(list(pre_tanh_mu_T.shape)) == 1, pre_tanh_mu_T
    assert len(list(std.shape)) == 1

    pre_tanh_mu_T.requires_grad_()
    tanh_mu_T = torch.tanh(pre_tanh_mu_T)

    # Get the upper bound of the Q estimate
    args = list(torch.unsqueeze(i, dim=0) for i in (ob, tanh_mu_T))
    Q1 = qfs[0](*args)
    Q2 = qfs[1](*args)

    mu_Q = (Q1 + Q2) / 2.0

    sigma_Q = torch.abs(Q1 - Q2) / 2.0

    Q_UB = mu_Q + beta_UB * sigma_Q

    # Obtain the gradient of Q_UB wrt to a
    # with a evaluated at mu_t
    grad = torch.autograd.grad(Q_UB, pre_tanh_mu_T)
    grad = grad[0]

    assert grad is not None
    assert pre_tanh_mu_T.shape == grad.shape

    # Obtain Sigma_T (the covariance of the normal distribution)
    Sigma_T = torch.pow(std, 2)

    # The dividor is (g^T Sigma g) ** 0.5
    # Sigma is diagonal, so this works out to be
    # ( sum_{i=1}^k (g^(i))^2 (sigma^(i))^2 ) ** 0.5
    denom = torch.sqrt(
        torch.sum(
            torch.mul(torch.pow(grad, 2), Sigma_T)
        )
    ) + 10e-6

    # Obtain the change in mu
    mu_C = math.sqrt(2.0 * delta) * torch.mul(Sigma_T, grad) / denom


    mu_E = pre_tanh_mu_T + mu_C

    # Construct the tanh normal distribution and sample the exploratory action from it
    assert mu_E.shape == std.shape

    dist = TanhNormal(mu_E, std)

    ac = dist.sample()

    ac_np = ptu.get_numpy(ac)
    args_new = list(torch.unsqueeze(i, dim=0) for i in (ob, ac))
    Q1_new = qfs[0](*args_new)
    Q2_new = qfs[1](*args_new)
    if Q2_new < Q2 or Q1_new < Q1:
        dist = TanhNormal(pre_tanh_mu_T, std)

        ac = dist.sample()

        ac_np = ptu.get_numpy(ac)
    return ac_np, {}
