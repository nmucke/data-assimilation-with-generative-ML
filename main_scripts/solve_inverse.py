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
from data_assimilation_with_generative_ML.transformer_layers import UNet_Tranformer

# set font size of plots
plt.rcParams.update({'axes.titlesize': 'small', 'axes.labelsize': 'small', 'xtick.labelsize': 'small', 'ytick.labelsize': 'small', 'legend.fontsize': 'small'})

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
    
    latent_vec = latent_vec.reshape(1, 1, gen_model.imsize, gen_model.imsize)

    pars_sample = ode_sampler(
        z=latent_vec,
        score_model=gen_model, 
        marginal_prob_std=marginal_prob_std_fn, 
        diffusion_coeff=diffusion_coeff_fn, 
        num_steps=15,
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
    
    observations = observations.flatten()
    
    # Compute the prior
    prior = torch.distributions.Normal(0, 1).log_prob(latent_vec).sum()

    # Compute the likelihood
    sample = sampling_model(latent_vec)
    pred_observation = observations_operator(sample)
    pred_observation = pred_observation.flatten()


    zero_mean = torch.zeros_like(pred_observation)
    likelihood = torch.distributions.Normal(zero_mean, 2*noise_std).log_prob(observations-pred_observation).sum()

    return prior + likelihood

def observation_operator(x):

    channels_obs_ids = torch.tensor([0, 2])
    horizontal_obs_ids = torch.tensor([25, 39])#torch.arange(4, 64, 8)
    vertical_obs_ids = torch.tensor([25, 39])#torch.arange(4, 64, 8)

    C, H, V = torch.meshgrid(channels_obs_ids, horizontal_obs_ids, vertical_obs_ids)

    x = x[0, C, H, V]

    # x = x[0, :, :, :, :]
    # x = x[0, horizontal_obs_ids, :, :]
    # x = x[:, vertical_obs_ids, :]

    return x

def main():

    data = xr.load_dataset(f'data/geodata/processed_DARTS_simulation_realization_1.nc')    

    perm_mean = -1.3754382634162903 #3.5748687267303465
    perm_std = 3.6160271644592283 #4.6395333366394045
    # por_mean = 0.09433708190917969
    # por_std = 0.03279830865561962

    pressure_mean = 336.26219848632815
    pressure_std = 130.7361669921875
    pressure_min = 51.5
    pressure_max = 299.43377685546875
    co2_mean = 0.03664950348436832
    co2_std = 0.13080736815929414
    U_z_min = -0.03506183251738548
    U_z_max = -7.1078920882428065e-06

    ft_min = 0.0
    ft_max = 28074.49609375


    perm_mean = -1.3754382634162903 #3.5748687267303465
    perm_std = 3.6160271644592283 #4.6395333366394045
    # self.por_mean = 0.09433708190917969
    # self.por_std = 0.03279830865561962


    pressure_mean = 59.839262017194635 # 336.26219848632815
    pressure_std = 16.946480998339133 #130.7361669921875
    co2_mean = 0.9997552563110158 #0.03664950348436832
    co2_std = 0.004887599662507033 #0.13080736815929414

    ft_min = 6.760696180663217e-08 #0.0
    ft_max = 0.0009333082125522196 #28074.49609375


    # state = np.stack((data['Pressure'].data, data['CO_2'].data), axis=0)
    state = np.concat((data['PRESSURE'].data, data['H2O'].data, data['U_z'].data), axis=1)
    pars = data['Perm'].data[0] # np.stack((data['Perm'].data, data['Por'].data), axis=0)
    ft = data['c_0_rate'].data[0]

    state = torch.tensor(state, dtype=torch.float32)
    state = torch.permute(state, (1, 2, 3, 0))
    pars = torch.tensor(pars, dtype=torch.float32)
    pars = torch.log(pars)
    # pars = pars.unsqueeze(0)
    ft = torch.tensor(ft, dtype=torch.float32)

    # state[0] = (state[0] - pressure_mean) / pressure_std
    state[0] = (state[0] - pressure_min) / (pressure_max - pressure_min)
    state[2] = (state[2] - U_z_min) / (U_z_max - U_z_min)

    # state[1] = (state[1] - co2_mean) / co2_std
    pars[0] = (pars[0] - perm_mean) / perm_std
    # pars[1] = (pars[1] - por_mean) / por_std

    ft = (ft - ft_min) / (ft_max - ft_min)

    # state = torch.permute(state, (0, 2, 3, 1))
    
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
        noise_std=noise_std
    )

    latent_vec = torch.randn(1, 1, 64, 64).to(device)
    latent_vec.requires_grad = True
    optimizer = torch.optim.Adam([latent_vec], lr=5e-2)

    for i in range(350):
        optimizer.zero_grad()
        loss = -log_prob(latent_vec)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Loss: {loss.item()}, Iteration: {i}')

    state_map = sampling_model(latent_vec)
    state_map_obs = observation_operator(state_map)

    latent_vec = latent_vec.reshape(1, 1, score_model.imsize, score_model.imsize)

    pars_map = ode_sampler(
        z=latent_vec,
        score_model=score_model, 
        marginal_prob_std=marginal_prob_std_fn, 
        diffusion_coeff=diffusion_coeff_fn, 
        num_steps=15,
        device=device
    )

    pres_min = state_true[0, 0].min().item()
    pres_max = state_true[0, 0].max().item()
    co2_min = state_true[0, 1].min().item()
    co2_max = state_true[0, 1].max().item()
    Uz_min = state_true[0, 2].min().item()
    Uz_max = state_true[0, 2].max().item()
    pars_min = pars[0, 0].min().item()
    pars_max = pars[0, 0].max().item()

    plt.figure()
    plt.subplot(5, 2, 1)
    plt.imshow(state_true.detach().cpu().numpy()[0, 0, :, :, -1], vmin=pres_min, vmax=pres_max)
    plt.scatter(X, Y, color='r', s=10)
    plt.title('True pressure')
    plt.colorbar()
    plt.subplot(5, 2, 2)
    plt.imshow(state_map.detach().cpu().numpy()[0, 0, :, :, -1], vmin=pres_min, vmax=pres_max)
    plt.colorbar()
    plt.title('MAP pressure')
    plt.subplot(5, 2, 3)
    plt.imshow(state_true.detach().cpu().numpy()[0, 1, :, :, -1], vmin=co2_min, vmax=co2_max)
    plt.title('True CO2')
    plt.colorbar()
    plt.subplot(5, 2, 4)
    plt.imshow(state_map.detach().cpu().numpy()[0, 1, :, :, -1], vmin=co2_min, vmax=co2_max)
    plt.colorbar()
    plt.title('MAP CO2')
    plt.subplot(5, 2, 5)
    plt.imshow(state_true.detach().cpu().numpy()[0, 2, :, :, -1], vmin=Uz_min, vmax=Uz_max)
    plt.title('True U_z')
    plt.colorbar()
    plt.subplot(5, 2, 6)
    plt.imshow(state_map.detach().cpu().numpy()[0, 2, :, :, -1], vmin=Uz_min, vmax=Uz_max)
    plt.colorbar()
    plt.subplot(5, 2, 7)
    plt.imshow(pars.detach().cpu().numpy()[0, 0, :, :], vmin=pars_min, vmax=pars_max)
    plt.colorbar()
    plt.title('True permiability')
    plt.subplot(5, 2, 8)
    plt.imshow(pars_map.detach().cpu().numpy()[0, 0, :, :], vmin=pars_min, vmax=pars_max)
    plt.colorbar()
    plt.title('MAP permiability')

    plt.subplot(5, 2, 9)
    plt.plot(state_obs_true.detach().cpu().numpy()[0, 0,0], label='True')
    plt.plot(state_obs.detach().cpu().numpy()[0, 0,0], label='True w. noise')
    plt.plot(state_map_obs.detach().cpu().numpy()[0, 0,0], label='MAP')
    plt.grid()
    plt.title('Observations')
    plt.legend()

    plt.subplot(5, 2, 10)
    plt.plot(state_obs_true.detach().cpu().numpy()[0, 1,1], label='True')
    plt.plot(state_obs.detach().cpu().numpy()[0, 1,1], label='True w. noise')
    plt.plot(state_map_obs.detach().cpu().numpy()[0, 1,1], label='MAP')
    plt.grid()
    plt.title('Observations')
    plt.legend()

    plt.savefig('inverse.png')
    
    plt.close()

    N = 100
    step_size = 1.0
    L = 5

    # HMC NUTS
    hamiltorch.set_random_seed(123)
    params_init = latent_vec.clone().flatten()#torch.randn(1, 1, 64, 64).to(device)
    params_init = params_init.to(device)
    burn = 50
    N_nuts = burn + N
    latent_hmc = hamiltorch.sample(
        log_prob_func=log_prob, 
        params_init=params_init,
        num_samples=N_nuts,
        step_size=step_size,
        num_steps_per_sample=L,
        sampler=hamiltorch.Sampler.HMC_NUTS, 
        burn=burn,
        desired_accept_rate=0.75
    )

    latent_hmc = torch.stack(latent_hmc)
    # latent_hmc = latent_hmc.reshape(N, 1, 64, 64)


    state_hmc = torch.zeros(N, 3, 64, 64, 120)

    with torch.no_grad():
        for i in range(N):
            state_hmc[i] = sampling_model(latent_hmc[i:i+1])[0]


    state_hmc_obs = torch.zeros(N, 2, 2, 2, 120)
    for i in range(N):
        state_hmc_obs[i] = observation_operator(state_hmc[i:i+1])

    state_hmc = torch.mean(state_hmc, dim=0)



    latent_hmc = latent_hmc.reshape(N, 1, 64, 64)
    hmc_x = ode_sampler(
        z=latent_hmc,
        score_model=score_model, 
        marginal_prob_std=marginal_prob_std_fn, 
        diffusion_coeff=diffusion_coeff_fn, 
        num_steps=25,
        device=device
    )

    pars_hmc = hmc_x.detach().cpu().numpy()
    pars_hmc = hmc_x[0:1]
    


    plt.figure(figsize=(15, 30))
    plt.tight_layout()

    plt.subplot(6, 3, 1)
    plt.imshow(state_true.detach().cpu().numpy()[0, 0, :, :, -1], vmin=pres_min, vmax=pres_max)
    plt.scatter(X, Y, color='r', s=10)
    plt.title('True pressure')
    plt.colorbar()
    plt.subplot(6, 3, 2)
    plt.imshow(state_map.detach().cpu().numpy()[0, 0, :, :, -1], vmin=pres_min, vmax=pres_max)
    plt.colorbar()
    plt.title('MAP pressure')
    plt.subplot(6, 3, 3)
    plt.imshow(state_hmc.detach().cpu().numpy()[0, :, :, -1], vmin=pres_min, vmax=pres_max)
    plt.title('Mean HMC pressure')
    plt.colorbar()



    plt.subplot(6, 3, 4)
    plt.imshow(state_true.detach().cpu().numpy()[0, 1, :, :, -1], vmin=co2_min, vmax=co2_max)
    plt.colorbar()
    plt.title('True H20')
    plt.subplot(6, 3, 5)
    plt.imshow(state_map.detach().cpu().numpy()[0, 1, :, :, -1], vmin=co2_min, vmax=co2_max)
    plt.colorbar()
    plt.title('MAP H20')
    plt.subplot(6, 3, 6)
    plt.imshow(state_hmc.detach().cpu().numpy()[1, :, :, -1], vmin=co2_min, vmax=co2_max)
    plt.colorbar()
    plt.title('Mean HMC H20')

    plt.subplot(6, 3, 7)
    plt.imshow(state_true.detach().cpu().numpy()[0, 2, :, :, -1], vmin=Uz_min, vmax=Uz_max)
    plt.scatter(X, Y, color='r', s=10)
    plt.colorbar()
    plt.title('True U_z')
    plt.subplot(6, 3, 8)
    plt.imshow(state_map.detach().cpu().numpy()[0, 2, :, :, -1], vmin=Uz_min, vmax=Uz_max)
    plt.colorbar()
    plt.title('MAP U_z')
    plt.subplot(6, 3, 9)
    plt.imshow(state_hmc.detach().cpu().numpy()[2, :, :, -1], vmin=Uz_min, vmax=Uz_max)
    plt.colorbar()
    plt.title('Mean HMC U_z')
    

    plt.subplot(6, 3, 10)
    plt.imshow(pars.detach().cpu().numpy()[0, 0, :, :], vmin=pars_min, vmax=pars_max)
    plt.colorbar()
    plt.title('True permiability')
    plt.subplot(6, 3, 11)
    plt.imshow(pars_map.detach().cpu().numpy()[0, 0, :, :], vmin=pars_min, vmax=pars_max)
    plt.colorbar()
    plt.title('MAP permiability')
    plt.subplot(6, 3, 12)
    plt.imshow(pars_hmc.detach().cpu().numpy()[0, 0, :, :], vmin=pars_min, vmax=pars_max)
    plt.colorbar()
    plt.title('Mean HMC permiability')

    plt.subplot(6, 3, 13)

    # plt.plot(state_obs.detach().cpu().numpy()[3,3], label='True w. noise')
    plt.plot(state_hmc_obs[0, 0, 0, 0], alpha=0.2, color='tab:red')
    for i in range(N):
        plt.plot(state_hmc_obs[i, 0, 0, 0], alpha=0.2, color='tab:red')
    plt.plot(state_hmc_obs.mean(dim=0).detach().cpu().numpy()[0, 0,0], label='HMC', color='tab:red', linewidth=4)
    plt.plot(state_map_obs.detach().cpu().numpy()[0, 0, 0], label='MAP', linewidth=4)
    plt.plot(state_obs_true.detach().cpu().numpy()[0, 0,0], label='True', linewidth=4)
    plt.grid()
    plt.title('Pressure observations')
    plt.legend()

    plt.subplot(6, 3, 14)
    # plt.plot(state_obs.detach().cpu().numpy()[5,5], label='True w. noise')
    plt.plot(state_hmc_obs[0, 0, 0, 1], alpha=0.2, color='tab:red')
    for i in range(N):
        plt.plot(state_hmc_obs[i,0,  0, 1], alpha=0.2, color='tab:red')
    plt.plot(state_hmc_obs.mean(dim=0).detach().cpu().numpy()[0, 0, 1], label='HMC', color='tab:red', linewidth=4)
    plt.plot(state_map_obs.detach().cpu().numpy()[0, 0, 1], label='MAP', linewidth=4)
    plt.plot(state_obs_true.detach().cpu().numpy()[0, 0, 1], label='True', linewidth=4)
    plt.grid()
    plt.title('Pressure observations')
    # plt.legend()


    plt.subplot(6, 3, 15)
    # plt.plot(state_obs.detach().cpu().numpy()[5,5], label='True w. noise')
    plt.plot(state_hmc_obs[0, 0, 1, 0], alpha=0.2, color='tab:red')
    for i in range(N):
        plt.plot(state_hmc_obs[i, 0,  1, 0], alpha=0.2, color='tab:red')
    plt.plot(state_hmc_obs.mean(dim=0).detach().cpu().numpy()[0, 1, 0], label='HMC', color='tab:red', linewidth=4)
    plt.plot(state_map_obs.detach().cpu().numpy()[0, 1, 0], label='MAP', linewidth=4)
    plt.plot(state_obs_true.detach().cpu().numpy()[0, 1, 0], label='True', linewidth=4)
    plt.grid()
    plt.title('Pressure observations')
    # plt.legend()


    plt.subplot(6, 3, 16)
    # plt.plot(state_obs.detach().cpu().numpy()[3,3], label='True w. noise')
    plt.plot(state_hmc_obs[0, 1, 0, 0], alpha=0.2, color='tab:red')
    for i in range(N):
        plt.plot(state_hmc_obs[i, 1, 0, 0], alpha=0.2, color='tab:red')
    plt.plot(state_hmc_obs.mean(dim=0).detach().cpu().numpy()[1, 0,0], label='HMC', color='tab:red', linewidth=4)
    plt.plot(state_map_obs.detach().cpu().numpy()[1, 0, 0], label='MAP', linewidth=4)
    plt.plot(state_obs_true.detach().cpu().numpy()[1, 0,0], label='True', linewidth=4)
    plt.grid()
    plt.title('Displacement observations')
    plt.legend()

    plt.subplot(6, 3, 17)
    # plt.plot(state_obs.detach().cpu().numpy()[5,5], label='True w. noise')
    plt.plot(state_hmc_obs[0, 1, 0, 1], alpha=0.2, color='tab:red')
    for i in range(N):
        plt.plot(state_hmc_obs[i,1,  0, 1], alpha=0.2, color='tab:red')
    plt.plot(state_hmc_obs.mean(dim=0).detach().cpu().numpy()[1, 0, 1], label='HMC', color='tab:red', linewidth=4)
    plt.plot(state_map_obs.detach().cpu().numpy()[1, 0, 1], label='MAP', linewidth=4)
    plt.plot(state_obs_true.detach().cpu().numpy()[1, 0, 1], label='True', linewidth=4)
    plt.grid()
    plt.title('Displacement observations')
    # plt.legend()


    plt.subplot(6, 3, 18)
    # plt.plot(state_obs.detach().cpu().numpy()[5,5], label='True w. noise')
    plt.plot(state_hmc_obs[0, 1, 1, 0], alpha=0.2, color='tab:red')
    for i in range(N):
        plt.plot(state_hmc_obs[i, 1,  1, 0], alpha=0.2, color='tab:red')
    plt.plot(state_hmc_obs.mean(dim=0).detach().cpu().numpy()[1, 1, 0], label='HMC', color='tab:red', linewidth=4)
    plt.plot(state_map_obs.detach().cpu().numpy()[1, 1, 0], label='MAP', linewidth=4)
    plt.plot(state_obs_true.detach().cpu().numpy()[1, 1, 0], label='True', linewidth=4)
    plt.grid()
    plt.title('Displacement observations')

    plt.savefig('inverse_HMC.pdf')
    
    plt.close()

if __name__ == "__main__":
    main()