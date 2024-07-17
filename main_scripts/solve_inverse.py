import functools
import pdb
import torch
import hamiltorch
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
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
from data_assimilation_with_generative_ML.variational_inference import compute_maximum_a_posteriori

import ray

hamiltorch.set_random_seed(123)

# set font size of plots
plt.rcParams.update({'axes.titlesize': 'small', 'axes.labelsize': 'small', 'xtick.labelsize': 'small', 'ytick.labelsize': 'small', 'legend.fontsize': 'small'})

hamiltorch.set_random_seed(123)
device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def latent_to_sample(
    latent_vec,
    gen_model,
    forward_model,
    marginal_prob_std_fn,
    diffusion_coeff_fn,
    ft,
    state_0,
    batch_size,
    device
):
    
    latent_vec = latent_vec.reshape(batch_size, 1, gen_model.imsize, gen_model.imsize)

    pars_sample = ode_sampler(
        z=latent_vec,
        score_model=gen_model, 
        marginal_prob_std=marginal_prob_std_fn, 
        diffusion_coeff=diffusion_coeff_fn, 
        num_steps=15,
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
    
    # observations = observations.flatten()
    
    # Compute the prior
    prior = torch.distributions.Normal(0, 1).log_prob(latent_vec).sum()

    # Compute the likelihood
    sample = sampling_model(latent_vec)
    pred_observation = observations_operator(sample)
    # pred_observation = pred_observation.flatten()

    observations = torch.tile(observations, (batch_size, 1, 1, 1, 1))

    zero_mean = torch.zeros_like(pred_observation)
    likelihood = torch.distributions.Normal(zero_mean, 2*noise_std).log_prob(observations-pred_observation).sum()

    return prior + likelihood

def observation_operator(x):

    channels_obs_ids = torch.tensor([0, 2])
    horizontal_obs_ids = torch.tensor([25, 39])#torch.arange(4, 64, 8)
    vertical_obs_ids = torch.tensor([25, 39])#torch.arange(4, 64, 8)

    C, H, V = torch.meshgrid(channels_obs_ids, horizontal_obs_ids, vertical_obs_ids)

    x = x[:, C, H, V]

    # x = x[0, :, :, :, :]
    # x = x[0, horizontal_obs_ids, :, :]
    # x = x[:, vertical_obs_ids, :]

    return x


# @ray.remote
def compute_posterior_samples(
    sampling_model,
    observation_operator,
    log_prob,
    device 
):
    latent_vec = torch.randn(1, 1, 64, 64).to(device)
    latent_map = compute_maximum_a_posteriori(
        latent_vec=latent_vec,
        sampling_model=sampling_model,
        observation_operator=observation_operator,
        log_prob=log_prob,
        num_iterations=11,
        lr=5e-2
    )

    return latent_map




# plot_map_results(
#     state_true=state_true,
#     state_map=state_map,
#     state_obs_true=state_obs_true,
#     state_obs=state_obs,
#     state_map_obs=state_map_obs,
#     pars=pars,
#     pars_map=pars_map,
#     X=X,
#     Y=Y
# ) 

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



def main():

    path = 'data/geodata/processed_DARTS_simulation_realization'
    dataset = ForwardModelDataset(path=path)
    state, pars, ft = dataset.__getitem__(10)
    
    state = state.unsqueeze(0)
    pars = pars.unsqueeze(0)
    ft = ft.unsqueeze(0)

    state = state.to(device)
    pars = pars.to(device)
    ft = ft.to(device)


    sigma =  25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    score_model = UNet_Tranformer(marginal_prob_std=marginal_prob_std_fn, imsize=64, in_channels=1)
    score_model.load_state_dict(torch.load('diffusion_model.pth'))
    score_model = score_model.to(device)
    score_model.eval()

    model_args = {
        'img_size': 64,
        'dim': 256,
        'patch_size': 8,
        'depth': 4,
        'heads': 8,
        'mlp_dim': 1024,
        'k': 128,
        'in_channels': 4,
    }
    forward_model = ForwardModel(model_args)
    forward_model.load_state_dict(torch.load('forward_model.pth'))
    forward_model = forward_model.to(device)
    forward_model.eval()
     

    num_chains = 8

    sampling_model = functools.partial(
        latent_to_sample, 
        gen_model=score_model, 
        forward_model=forward_model,
        marginal_prob_std_fn=marginal_prob_std_fn, 
        diffusion_coeff_fn=diffusion_coeff_fn, 
        device=device,
        ft=ft,
        state_0=state[:, :, :, :, 0],
        batch_size=num_chains
    )

    noise_std = 0.01
    state_true = state.clone()
    state_noisy = state_true + torch.randn_like(state_true) * noise_std
    state_obs = observation_operator(state_noisy)
    state_obs = state_obs.to(device)

    state_obs_true = observation_operator(state_true)

    X, Y = np.meshgrid(np.array([25, 39]), np.array([25, 39]))


    log_prob = functools.partial(
        log_posterior, 
        observations=state_obs, 
        observations_operator=observation_operator, 
        sampling_model=sampling_model, 
        noise_std=noise_std,
        batch_size=num_chains
    )


    N_samples_pr_chain = 10
    N = N_samples_pr_chain * num_chains
    burn = 10
    step_size = 1.0
    L = 5
    latent_hmc = []
    latent_map = []
    # ray.init(num_cpus=num_chains)
    # for i in range(num_chains):

    #     latent_map_i = compute_posterior_samples.remote(
    #         sampling_model=sampling_model,
    #         observation_operator=observation_operator,
    #         log_prob=log_prob,
    #         device=device
    #     )

    #     latent_map.append(latent_map_i)

    # latent_map = ray.get(latent_map)
    # latent_map = torch.stack(latent_map)


    latent_vec = torch.randn(num_chains, 1, 64, 64).to(device)
    latent_map = compute_maximum_a_posteriori(
        latent_vec=latent_vec,
        sampling_model=sampling_model,
        observation_operator=observation_operator,
        log_prob=log_prob,
        num_iterations=300,
        lr=5e-2
    )

    state_map = sampling_model(latent_map)
    state_map_obs = observation_operator(state_map)

    latent_vec = latent_vec.reshape(num_chains, 1, score_model.imsize, score_model.imsize)

    pars_map = ode_sampler(
        z=latent_map,
        score_model=score_model, 
        marginal_prob_std=marginal_prob_std_fn, 
        diffusion_coeff=diffusion_coeff_fn, 
        num_steps=15,
        device=device
    )

    sampling_model = functools.partial(
        latent_to_sample, 
        gen_model=score_model.to('cpu'), 
        forward_model=forward_model.to('cpu'),
        marginal_prob_std_fn=marginal_prob_std_fn, 
        diffusion_coeff_fn=diffusion_coeff_fn, 
        device='cpu',
        ft=ft.to('cpu'),
        state_0=state[:, :, :, :, 0].to('cpu'),
        batch_size=1
    )
    log_prob = functools.partial(
        log_posterior, 
        observations=state_obs.to('cpu'), 
        observations_operator=observation_operator, 
        sampling_model=sampling_model, 
        noise_std=noise_std,
        batch_size=1
    )

     # HMC NUTS
    # params_init = latent_vec.clone().flatten()#torch.randn(1, 1, 64, 64).to(device)
    # params_init = params_init.to(device)
    # N_nuts = burn + N_samples_pr_chain
    # latent_hmc_chain = hamiltorch.sample(
    #     log_prob_func=log_prob, 
    #     params_init=params_init,
    #     num_samples=N_nuts,
    #     step_size=step_size,
    #     num_steps_per_sample=L,
    #     sampler=hamiltorch.Sampler.HMC_NUTS, 
    #     burn=burn,
    #     desired_accept_rate=0.80
    # )

    # latent_hmc_chain = torch.stack(latent_hmc_chain)
    # # latent_hmc = latent_hmc.reshape(N, 1, 64, 64)

    ray.init(num_cpus=num_chains)
    latent_hmc = []
    for i in range(num_chains):
        latent_hmc_chain = compute_HMC_samples.remote(
            latent_vec=latent_vec[i:i+1].to('cpu'),
            log_prob=log_prob,
            N_samples_pr_chain=N_samples_pr_chain,
            burn=burn,
            step_size=step_size,
            L=L,
            device='cpu'
        )

        latent_hmc.append(latent_hmc_chain)

    latent_hmc = ray.get(latent_hmc)
    latent_hmc = torch.stack(latent_hmc)

    latent_hmc = latent_hmc.reshape(N, 1, 64, 64)

    state_hmc = torch.zeros(N, 3, 64, 64, 120)

    with torch.no_grad():
        for i in range(N):
            state_hmc[i] = sampling_model(latent_hmc[i:i+1])[0]

    state_hmc_obs = torch.zeros(N, 2, 2, 2, 120)
    for i in range(N):
        state_hmc_obs[i] = observation_operator(state_hmc[i:i+1])

    state_hmc = torch.mean(state_hmc, dim=0)

    latent_hmc = latent_hmc.reshape(N, 1, 64, 64)
    pars_hmc = ode_sampler(
        z=latent_hmc.to(device),
        score_model=score_model.to(device), 
        marginal_prob_std=marginal_prob_std_fn, 
        diffusion_coeff=diffusion_coeff_fn, 
        num_steps=15,
        device=device
    )

    pars_hmc = pars_hmc.mean(axis=0).unsqueeze(0)

    
    plot_monte_carlo_results(
        state_true=state_true,
        state_map=state_map[0:1],
        state_monte_carlo=state_hmc,
        state_obs_true=state_obs_true[0],
        state_map_obs=state_map_obs[0],
        state_monte_carlo_obs=state_hmc_obs,
        pars=pars,
        pars_map=pars_map,
        pars_monte_carlo=pars_hmc,
        X=X,
        Y=Y
    )

if __name__ == "__main__":
    main()