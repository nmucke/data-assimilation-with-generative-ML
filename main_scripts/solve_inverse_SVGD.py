import functools
import pdb
import torch
import torch.nn as nn
import hamiltorch
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from data_assimilation_with_generative_ML.bayesian_inference.hamiltonian_monte_carlo import compute_HMC_samples
from data_assimilation_with_generative_ML.bayesian_inference.stein_variational_gradient_descent import compute_SVGD
from data_assimilation_with_generative_ML.datasets import ForwardModelDataset
from data_assimilation_with_generative_ML.diffusion_models import (
    DiTScoreNet,
    loss_fn,
    marginal_prob_std,
    diffusion_coeff,
    pc_sampler,
    ode_sampler,
)
from data_assimilation_with_generative_ML.forward_models import ForwardModel
from data_assimilation_with_generative_ML.plotting_utils import plot_map_results, plot_monte_carlo_results
from data_assimilation_with_generative_ML.transformer_layers import UNet_Tranformer
from data_assimilation_with_generative_ML.bayesian_inference.variational_inference import compute_maximum_a_posteriori

import ray

from data_assimilation_with_generative_ML.bayesian_inference.utils import latent_to_state, log_posterior, observation_operator

hamiltorch.set_random_seed(123)

# set font size of plots
plt.rcParams.update({'axes.titlesize': 'small', 'axes.labelsize': 'small', 'xtick.labelsize': 'small', 'ytick.labelsize': 'small', 'legend.fontsize': 'small'})

hamiltorch.set_random_seed(123)
device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main():

    path = 'data/geodata/processed_DARTS_simulation_realization'

    dataset = ForwardModelDataset(path=path)
    state, pars, ft = dataset.__getitem__(50)
    
    state = state.unsqueeze(0)
    pars = pars.unsqueeze(0)
    ft = ft.unsqueeze(0)

    state = state.to(device)
    pars = pars.to(device)
    ft = ft.to(device)


    sigma =  25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    pars_model_args = {
        'marginal_prob_std': marginal_prob_std_fn, 
        'in_channels': 1,
        'channels': [16, 32, 64, 128], # [4, 8, 16, 32]
        'imsize': 64
    }
    score_model = UNet_Tranformer(**pars_model_args)
    score_model.load_state_dict(torch.load('diffusion_model.pth'))
    score_model = score_model.to(device)
    score_model.eval()

    model_args = {
        'marginal_prob_std': None, 
        'in_channels': 4,
        'channels': [8, 16, 32, 64],
        'imsize': 64,
    }
    forward_model = ForwardModel(model_args)
    forward_model.load_state_dict(torch.load('forward_model.pth'))
    forward_model = forward_model.to(device)
    forward_model.eval()

    N = 12


    noise_std = 0.01
    state_true = state.clone()
    state_noisy = state_true + torch.randn_like(state_true) * noise_std
    state_obs = observation_operator(state_noisy)
    state_obs = state_obs.to(device)

    state_obs_true = observation_operator(state_true)

    # x = np.array([25, 39])
    # y = np.array([25, 39])
    x = torch.arange(4, 64, 8)
    y = torch.arange(4, 64, 8)

    X, Y = np.meshgrid(x, y)
    # X, Y = np.meshgrid(torch.arange(4, 64, 8), torch.arange(4, 64, 8))

    num_obs_X = len(x)
    num_obs_Y = len(y)

    sampling_model = functools.partial(
        latent_to_state, 
        gen_model=score_model, 
        forward_model=forward_model,
        marginal_prob_std_fn=marginal_prob_std_fn, 
        diffusion_coeff_fn=diffusion_coeff_fn, 
        device=device,
        ft=ft,
        state_0=state[:, :, :, :, 0],
        batch_size=1,
        num_generative_steps=10
    )
    log_prob = functools.partial(
        log_posterior, 
        observations=state_obs, 
        observations_operator=observation_operator, 
        sampling_model=sampling_model, 
        noise_std=noise_std,
        batch_size=1
    )

    latent_map = torch.randn(1, 1, 64, 64).to(device)

    latent_map = compute_maximum_a_posteriori(
        latent_vec=latent_map,
        log_prob=log_prob,
        num_iterations=5,
        lr=1e0
    )

    state_map = sampling_model(latent_map)
    state_map_obs = observation_operator(state_map)

    latent_map = latent_map.reshape(1, 1, score_model.imsize, score_model.imsize)

    pars_map = ode_sampler(
        z=latent_map,
        score_model=score_model,    
        marginal_prob_std=marginal_prob_std_fn, 
        diffusion_coeff=diffusion_coeff_fn, 
        num_steps=15,
        device=device
    )


    sampling_model = functools.partial(
        latent_to_state, 
        gen_model=score_model, 
        forward_model=forward_model,
        marginal_prob_std_fn=marginal_prob_std_fn, 
        diffusion_coeff_fn=diffusion_coeff_fn, 
        device=device,
        ft=ft,
        state_0=state[:, :, :, :, 0],
        batch_size=N,
        num_generative_steps=10
    )
    log_prob = functools.partial(
        log_posterior, 
        observations=state_obs, 
        observations_operator=observation_operator, 
        sampling_model=sampling_model, 
        noise_std=noise_std,
        batch_size=N
    )
    
    latent_vec = torch.randn(N, 1, 64, 64).to(device)
    latent_vec = nn.Flatten()(latent_vec)
    latent_svgd = compute_SVGD(
        latent_vec=latent_vec,
        log_prob=log_prob,
        num_iterations=100,
        lr=5e-2,
    )

    latent_svgd = latent_svgd.reshape(N, 1, 64, 64)

    with torch.no_grad():
        state_svgd = sampling_model(latent_svgd)

    state_svgd_obs = torch.zeros(N, 2, num_obs_X, num_obs_X, 121)
    for i in range(N):
        state_svgd_obs[i] = observation_operator(state_svgd[i:i+1])

    state_svgd = torch.mean(state_svgd, dim=0)

    latent_svgd = latent_svgd.reshape(N, 1, 64, 64)
    pars_svgd = ode_sampler(
        z=latent_svgd.to(device),
        score_model=score_model.to(device), 
        marginal_prob_std=marginal_prob_std_fn, 
        diffusion_coeff=diffusion_coeff_fn, 
        num_steps=15,
        device=device
    )

    pars_svgd = pars_svgd.mean(axis=0).unsqueeze(0)

    plot_monte_carlo_results(
        state_true=state_true,
        state_map=state_map[0:1],
        state_monte_carlo=state_svgd,
        state_obs_true=state_obs_true[0],
        state_map_obs=state_map_obs[0],
        state_monte_carlo_obs=state_svgd_obs,
        pars=pars,
        pars_map=pars_map,
        pars_monte_carlo=pars_svgd,
        X=X,
        Y=Y,
        plot_path='svgd_inverse'
    )

if __name__ == "__main__":
    main()