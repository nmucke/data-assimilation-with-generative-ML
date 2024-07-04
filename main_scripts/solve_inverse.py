import functools
import pdb
import torch
import hamiltorch
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from data_assimilation_with_generative_ML.diffusion_models import (
    DiTScoreNet,
    loss_fn,
    marginal_prob_std,
    diffusion_coeff,
    pc_sampler,
    ode_sampler,
)
from data_assimilation_with_generative_ML.forward_models import ForwardModel

hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def latent_to_sample(
    latent_vec,
    gen_model,
    forward_model,
    marginal_prob_std_fn,
    diffusion_coeff_fn,
    ft,
    state_0,
    device
):
    
    latent_vec = latent_vec.reshape(1, 2, gen_model.imsize, gen_model.imsize)

    pars_sample = ode_sampler(
        z=latent_vec,
        score_model=gen_model, 
        marginal_prob_std=marginal_prob_std_fn, 
        diffusion_coeff=diffusion_coeff_fn, 
        num_steps=50,
        device=device
    )

    sample = forward_model(state_0, pars_sample, ft)

    return sample

def log_posterior(
    latent_vec: torch.Tensor,
    observations: torch.Tensor,
    observations_operator,
    sampling_model,
    noise_std: float
):
    
    # Compute the prior
    prior = torch.distributions.Normal(0, 1).log_prob(latent_vec).sum()

    # Compute the likelihood
    sample = sampling_model(latent_vec)
    pred_observation = observations_operator(sample)

    zero_mean = torch.zeros_like(pred_observation)
    likelihood = torch.distributions.Normal(zero_mean, noise_std).log_prob(observations-pred_observation).sum()

    return prior + likelihood

def observation_operator(x):

    horizontal_obs_ids = torch.arange(4, 64, 8)
    vertical_obs_ids = torch.arange(4, 64, 8)

    x = x[0, 0, horizontal_obs_ids, :, :]
    x = x[:, vertical_obs_ids, :]

    return x

def main():

    data = xr.load_dataset(f'data/results64/simulation_results_realization_64x64_1.nc')     

    perm_mean = 3.5748687267303465
    perm_std = 4.6395333366394045
    por_mean = 0.09433708190917969
    por_std = 0.03279830865561962

    pressure_mean = 336.26219848632815
    pressure_std = 130.7361669921875
    co2_mean = 0.03664950348436832
    co2_std = 0.13080736815929414

    ft_min = 0.0
    ft_max = 28074.49609375

    state = np.stack((data['Pressure'].data, data['CO_2'].data), axis=0)
    pars = np.stack((data['Perm'].data, data['Por'].data), axis=0)
    ft = data['gas_rate'].data[0]

    state = torch.tensor(state, dtype=torch.float32)
    pars = torch.tensor(pars, dtype=torch.float32)
    ft = torch.tensor(ft, dtype=torch.float32)

    state[0] = (state[0] - pressure_mean) / pressure_std
    state[1] = (state[1] - co2_mean) / co2_std
    pars[0] = (pars[0] - perm_mean) / perm_std
    pars[1] = (pars[1] - por_mean) / por_std

    ft = (ft - ft_min) / (ft_max - ft_min)

    state = torch.permute(state, (0, 2, 3, 1))
    
    state = state.unsqueeze(0)
    pars = pars.unsqueeze(0)
    ft = ft.unsqueeze(0)

    state = state.to(device)
    pars = pars.to(device)
    ft = ft.to(device)


    sigma =  25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    score_model = DiTScoreNet(marginal_prob_std=marginal_prob_std_fn, imsize=64, in_channels=2)
    score_model.load_state_dict(torch.load('ckpt.pth'))
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

    sampling_model = functools.partial(
        latent_to_sample, 
        gen_model=score_model, 
        forward_model=forward_model,
        marginal_prob_std_fn=marginal_prob_std_fn, 
        diffusion_coeff_fn=diffusion_coeff_fn, 
        device=device,
        ft=ft,
        state_0=state[:, :, :, :, 0],
    )

    noise_std = 0.1
    state_true = state.clone()
    state_noisy = state_true# + torch.randn_like(state_true) * noise_std
    state_obs = observation_operator(state_noisy)
    state_obs = state_obs.to(device)

    state_obs_true = observation_operator(state_true)

    X, Y = np.meshgrid(np.arange(4, 64, 8), np.arange(4, 64, 8))

    log_prob = functools.partial(
        log_posterior, 
        observations=state_obs, 
        observations_operator=observation_operator, 
        sampling_model=sampling_model, 
        noise_std=noise_std
    )

    latent_vec = torch.randn(1, 2, 64, 64).to(device)
    latent_vec.requires_grad = True
    optimizer = torch.optim.Adam([latent_vec], lr=5e-3)

    for i in range(10):
        optimizer.zero_grad()
        loss = -log_prob(latent_vec)
        loss.backward()
        optimizer.step()
        print(f'Loss: {loss.item()}, Iteration: {i}')

    state_map = sampling_model(latent_vec)
    state_map_obs = observation_operator(state_map)

    latent_vec = latent_vec.reshape(1, 2, score_model.imsize, score_model.imsize)

    pars_map = ode_sampler(
        z=latent_vec,
        score_model=score_model, 
        marginal_prob_std=marginal_prob_std_fn, 
        diffusion_coeff=diffusion_coeff_fn, 
        num_steps=50,
        device=device
    )


    plt.figure()
    plt.subplot(3, 2, 1)
    plt.imshow(state_true.detach().cpu().numpy()[0, 0, :, :, -1])
    plt.scatter(X, Y, color='r', s=10)
    plt.title('True state')
    plt.colorbar()
    plt.subplot(3, 2, 2)
    plt.imshow(state_map.detach().cpu().numpy()[0, 0, :, :, -1])
    # plt.scatter(X, Y, color='r', s=10)
    plt.colorbar()
    plt.title('MAP state')
    plt.subplot(3, 2, 3)
    plt.imshow(pars.detach().cpu().numpy()[0, 0, :, :])
    plt.colorbar()
    plt.title('True permiability')
    plt.subplot(3, 2, 4)
    plt.imshow(pars_map.detach().cpu().numpy()[0, 0, :, :])
    plt.colorbar()
    plt.title('MAP permiability')

    plt.subplot(3, 2, 5)
    plt.plot(state_obs_true.detach().cpu().numpy()[0,0], label='True')
    plt.plot(state_obs.detach().cpu().numpy()[0,0], label='True w. noise')
    plt.plot(state_map_obs.detach().cpu().numpy()[0,0], label='MAP')
    plt.grid()
    plt.title('Observations')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(state_obs_true.detach().cpu().numpy()[3,3], label='True')
    plt.plot(state_obs.detach().cpu().numpy()[3,3], label='True w. noise')
    plt.plot(state_map_obs.detach().cpu().numpy()[3,3], label='MAP')
    plt.grid()
    plt.title('Observations')
    plt.legend()
    
    plt.show()

    # N = 50
    # step_size = .3
    # L = 5

    # # HMC NUTS
    # hamiltorch.set_random_seed(123)
    # params_init = latent_vec.flatten()#torch.randn(1, 1, 64, 64).to(device)
    # burn=50
    # N_nuts = burn + N
    # latent_hmc = hamiltorch.sample(
    #     log_prob_func=log_prob, 
    #     params_init=params_init,
    #     num_samples=N_nuts,
    #     step_size=step_size,
    #     num_steps_per_sample=L,
    #     sampler=hamiltorch.Sampler.HMC_NUTS, 
    #     burn=burn,
    #     desired_accept_rate=0.8
    # )

    # latent_hmc = torch.stack(latent_hmc)
    # latent_hmc = latent_hmc.reshape(N, 1, 64, 64)


    # hmc_x = ode_sampler(
    #     z=latent_vec,
    #     score_model=score_model, 
    #     marginal_prob_std=marginal_prob_std_fn, 
    #     diffusion_coeff=diffusion_coeff_fn, 
    #     num_steps=25,
    #     device=device
    # )
    # hmc_x = hmc_x.detach().cpu().numpy()
    # hmc_x = hmc_x.mean(axis=0)



    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.imshow(x_true.cpu().numpy()[0, 0])
    # plt.scatter(X, Y, color='r', s=10)
    # plt.colorbar()
    # plt.title('True observation')
    # plt.subplot(1, 3, 2)
    # plt.imshow(map_x.detach().cpu().numpy()[0, 0])
    # plt.colorbar()
    # plt.title('Max Posterior')
    # plt.subplot(1, 3, 3)
    # plt.imshow(hmc_x[0])
    # plt.colorbar()
    # plt.title('HMC')
    # plt.show()

if __name__ == "__main__":
    main()