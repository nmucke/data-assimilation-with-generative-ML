import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import hamiltorch
import matplotlib.pyplot as plt
import ray


@ray.remote
def compute_HMC_samples(
    latent_vec,
    log_prob,
    N_samples_pr_chain,
    burn,
    step_size,
    L,
    device
):

    # HMC NUTS
    params_init = latent_vec.clone().flatten()#torch.randn(1, 1, 64, 64).to(device)
    params_init = params_init.to(device)
    N_nuts = burn + N_samples_pr_chain
    latent_hmc_chain = hamiltorch.sample(
        log_prob_func=log_prob, 
        params_init=params_init,
        num_samples=N_nuts,
        step_size=step_size,
        num_steps_per_sample=L,
        sampler=hamiltorch.Sampler.HMC_NUTS, 
        burn=burn,
        desired_accept_rate=0.80
    )

    latent_hmc_chain = torch.stack(latent_hmc_chain)
    # latent_hmc = latent_hmc.reshape(N, 1, 64, 64)

    return latent_hmc_chain

