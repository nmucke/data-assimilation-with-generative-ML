""" Baed on https://github.com/activatedgeek/svgd/tree/master """
import pdb
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np


class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()

        self.sigma = sigma

    def forward(self, X, Y):

        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY
  

class SVGD:
    def __init__(self, log_prob, kernel, optimizer):
        self.log_prob = log_prob
        self.kernel = kernel
        self.optim = optimizer

    def phi(self, X):
        X = X.detach().requires_grad_(True)

        log_prob = self.log_prob(X)
        score_func = autograd.grad(log_prob.sum(), X)[0]

        K_XX = self.kernel(X, X.detach())
        grad_K = -autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

        return phi

    def step(self, X):
        self.optim.zero_grad()
        X.grad = -self.phi(X)
        self.optim.step()
    

def compute_SVGD(
    latent_vec,
    log_prob,
    num_iterations=500,
    lr=5e-2,
    sigma=None
):
    optimizer = torch.optim.Adam([latent_vec], lr=lr)
    rbf_kernel = RBF(sigma=sigma)
    svgd = SVGD(log_prob=log_prob, kernel=rbf_kernel, optimizer=optimizer)

    for i in range(num_iterations):
        svgd.step(latent_vec)

        if i % 10 == 0:
            print(f'Iteration: {i}')

    return latent_vec