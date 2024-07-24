import torch
from langevin_sampling.samplers import LangevinDynamics
import numpy as np
import matplotlib.pyplot as plt
from rosenbrock import rosenbrock
import copy
from tqdm import tqdm



def compute_langevin_samples(
    log_prob,
    latent_vec,
    num_iterations=500,
    lr=1e-1,
    burn_in=500,
    device='cpu'
):

    langevin_dynamics = LangevinDynamics(
        x=latent_vec,
        func= lambda x: -log_prob(x),
        lr=1e-1,
        lr_final=1e-2,
        max_itr=num_iterations,
        device=device
    )

    hist_samples = []
    loss_log = []
    for j in tqdm(range(burn_in + num_iterations)):
        est, loss = langevin_dynamics.sample()
        loss_log.append(loss)

        # if j % 10 == 0:
        hist_samples.append(est.cpu().detach())

        if j % 1 == 0:
            print(f'Loss: {loss}, Iteration: {j}')

    samples = torch.stack(hist_samples)[burn_in:]

    return samples 

