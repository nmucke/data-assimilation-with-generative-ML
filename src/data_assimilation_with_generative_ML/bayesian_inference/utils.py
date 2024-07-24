import pdb
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

from data_assimilation_with_generative_ML.diffusion_models import ode_sampler



def latent_to_state(
    latent_vec,
    gen_model,
    forward_model,
    marginal_prob_std_fn,
    diffusion_coeff_fn,
    ft,
    state_0,
    batch_size,
    num_generative_steps,
    device
):
    
    latent_vec = latent_vec.reshape(batch_size, 1, gen_model.imsize, gen_model.imsize)

    pars_sample = ode_sampler(
        z=latent_vec,
        score_model=gen_model, 
        marginal_prob_std=marginal_prob_std_fn, 
        diffusion_coeff=diffusion_coeff_fn, 
        num_steps=num_generative_steps,
        device=device
    )

    state_0 = torch.tile(state_0, (batch_size, 1, 1, 1))
    ft = torch.tile(ft, (batch_size, 1))


    sample = forward_model(state_0, pars_sample, ft)

    return sample

def log_posterior(
    latent_vec: torch.Tensor,
    observations: torch.Tensor,
    observations_operator,
    sampling_model,
    noise_std: float,
    batch_size: int = 1
):
    
    # Compute the prior

    # norm of each latent vector
    prior = 0
    for i in range(batch_size):
        # latent_norm = torch.norm(latent_vec[i])
        # prior += torch.distributions.Normal(0, 1).log_prob(latent_norm).sum()
        prior += torch.distributions.Normal(0, 1).log_prob(latent_vec[i]).sum()

    # Compute the likelihood
    sample = sampling_model(latent_vec)

    pred_observation = observations_operator(sample)
    # pred_observation = pred_observation.flatten()

    # observations = torch.tile(observations, (batch_size, 1, 1, 1, 1))

    zero_mean = torch.zeros_like(observations)

    likelihood = 0
    for i in range(batch_size):
        likelihood += torch.distributions.Normal(zero_mean, 2*noise_std).log_prob(observations[0]-pred_observation[i]).sum()
    # likelihood = torch.distributions.Normal(zero_mean, 2*noise_std).log_prob(observations-pred_observation).sum()

    return prior + likelihood

def observation_operator(x):

    channels_obs_ids = torch.tensor([0, 2])
    # horizontal_obs_ids = torch.tensor([25, 39])
    # vertical_obs_ids = torch.tensor([25, 39])
    horizontal_obs_ids = torch.arange(4, 64, 8) #
    vertical_obs_ids = torch.arange(4, 64, 8) #

    C, H, V = torch.meshgrid(channels_obs_ids, horizontal_obs_ids, vertical_obs_ids)

    x = x[:, C, H, V]

    return x