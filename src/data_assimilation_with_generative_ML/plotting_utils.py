
import matplotlib.pyplot as plt

def plot_map_results(
    state_true, 
    state_map, 
    state_obs_true, 
    state_obs, 
    state_map_obs, 
    pars, 
    pars_map, 
    X, 
    Y
):

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



def plot_monte_carlo_results(
    state_true,
    state_map,
    state_monte_carlo,
    state_obs_true,
    state_map_obs,
    state_monte_carlo_obs,
    pars,
    pars_map,
    pars_monte_carlo,
    X,
    Y,
):
    
    pres_min = state_true[0, 0].min().item()
    pres_max = state_true[0, 0].max().item()
    co2_min = state_true[0, 1].min().item()
    co2_max = state_true[0, 1].max().item()
    Uz_min = state_true[0, 2].min().item()
    Uz_max = state_true[0, 2].max().item()
    pars_min = pars[0, 0].min().item()
    pars_max = pars[0, 0].max().item()

    N = state_monte_carlo.shape[0]


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
    plt.imshow(state_monte_carlo.detach().cpu().numpy()[0, :, :, -1], vmin=pres_min, vmax=pres_max)
    plt.title('Mean monte_carlo pressure')
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
    plt.imshow(state_monte_carlo.detach().cpu().numpy()[1, :, :, -1], vmin=co2_min, vmax=co2_max)
    plt.colorbar()
    plt.title('Mean monte_carlo H20')

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
    plt.imshow(state_monte_carlo.detach().cpu().numpy()[2, :, :, -1], vmin=Uz_min, vmax=Uz_max)
    plt.colorbar()
    plt.title('Mean monte_carlo U_z')
    

    plt.subplot(6, 3, 10)
    plt.imshow(pars.detach().cpu().numpy()[0, 0, :, :], vmin=pars_min, vmax=pars_max)
    plt.colorbar()
    plt.title('True permiability')
    plt.subplot(6, 3, 11)
    plt.imshow(pars_map.detach().cpu().numpy()[0, 0, :, :], vmin=pars_min, vmax=pars_max)
    plt.colorbar()
    plt.title('MAP permiability')
    plt.subplot(6, 3, 12)
    plt.imshow(pars_monte_carlo.detach().cpu().numpy()[0, 0, :, :], vmin=pars_min, vmax=pars_max)
    plt.colorbar()
    plt.title('Mean monte_carlo permiability')

    plt.subplot(6, 3, 13)

    # plt.plot(state_obs.detach().cpu().numpy()[3,3], label='True w. noise')
    plt.plot(state_monte_carlo_obs[0, 0, 0, 0], alpha=0.2, color='tab:red')
    for i in range(N):
        plt.plot(state_monte_carlo_obs[i, 0, 0, 0], alpha=0.2, color='tab:red')
    plt.plot(state_monte_carlo_obs.mean(dim=0).detach().cpu().numpy()[0, 0,0], label='monte_carlo', color='tab:red', linewidth=4)
    plt.plot(state_map_obs.detach().cpu().numpy()[0, 0, 0], label='MAP', linewidth=4)
    plt.plot(state_obs_true.detach().cpu().numpy()[0, 0,0], label='True', linewidth=4)
    plt.grid()
    plt.title('Pressure observations')
    plt.legend()

    plt.subplot(6, 3, 14)
    # plt.plot(state_obs.detach().cpu().numpy()[5,5], label='True w. noise')
    plt.plot(state_monte_carlo_obs[0, 0, 0, 1], alpha=0.2, color='tab:red')
    for i in range(N):
        plt.plot(state_monte_carlo_obs[i, 0, 0, 1], alpha=0.2, color='tab:red')
    plt.plot(state_monte_carlo_obs.mean(dim=0).detach().cpu().numpy()[0, 0, 1], label='monte_carlo', color='tab:red', linewidth=4)
    plt.plot(state_map_obs.detach().cpu().numpy()[0, 0, 1], label='MAP', linewidth=4)
    plt.plot(state_obs_true.detach().cpu().numpy()[0, 0, 1], label='True', linewidth=4)
    plt.grid()
    plt.title('Pressure observations')
    # plt.legend()


    plt.subplot(6, 3, 15)
    # plt.plot(state_obs.detach().cpu().numpy()[5,5], label='True w. noise')
    plt.plot(state_monte_carlo_obs[0, 0, 1, 0], alpha=0.2, color='tab:red')
    for i in range(N):
        plt.plot(state_monte_carlo_obs[i, 0,  1, 0], alpha=0.2, color='tab:red')
    plt.plot(state_monte_carlo_obs.mean(dim=0).detach().cpu().numpy()[0, 1, 0], label='monte_carlo', color='tab:red', linewidth=4)
    plt.plot(state_map_obs.detach().cpu().numpy()[0, 1, 0], label='MAP', linewidth=4)
    plt.plot(state_obs_true.detach().cpu().numpy()[0, 1, 0], label='True', linewidth=4)
    plt.grid()
    plt.title('Pressure observations')
    # plt.legend()


    plt.subplot(6, 3, 16)
    # plt.plot(state_obs.detach().cpu().numpy()[3,3], label='True w. noise')
    plt.plot(state_monte_carlo_obs[0, 1, 0, 0], alpha=0.2, color='tab:red')
    for i in range(N):
        plt.plot(state_monte_carlo_obs[i, 1, 0, 0], alpha=0.2, color='tab:red')
    plt.plot(state_monte_carlo_obs.mean(dim=0).detach().cpu().numpy()[1, 0,0], label='monte_carlo', color='tab:red', linewidth=4)
    plt.plot(state_map_obs.detach().cpu().numpy()[1, 0, 0], label='MAP', linewidth=4)
    plt.plot(state_obs_true.detach().cpu().numpy()[1, 0,0], label='True', linewidth=4)
    plt.grid()
    plt.title('Displacement observations')
    plt.legend()

    plt.subplot(6, 3, 17)
    # plt.plot(state_obs.detach().cpu().numpy()[5,5], label='True w. noise')
    plt.plot(state_monte_carlo_obs[0, 1, 0, 1], alpha=0.2, color='tab:red')
    for i in range(N):
        plt.plot(state_monte_carlo_obs[i,1,  0, 1], alpha=0.2, color='tab:red')
    plt.plot(state_monte_carlo_obs.mean(dim=0).detach().cpu().numpy()[1, 0, 1], label='monte_carlo', color='tab:red', linewidth=4)
    plt.plot(state_map_obs.detach().cpu().numpy()[1, 0, 1], label='MAP', linewidth=4)
    plt.plot(state_obs_true.detach().cpu().numpy()[1, 0, 1], label='True', linewidth=4)
    plt.grid()
    plt.title('Displacement observations')
    # plt.legend()


    plt.subplot(6, 3, 18)
    # plt.plot(state_obs.detach().cpu().numpy()[5,5], label='True w. noise')
    plt.plot(state_monte_carlo_obs[0, 1, 1, 0], alpha=0.2, color='tab:red')
    for i in range(N):
        plt.plot(state_monte_carlo_obs[i, 1,  1, 0], alpha=0.2, color='tab:red')
    plt.plot(state_monte_carlo_obs.mean(dim=0).detach().cpu().numpy()[1, 1, 0], label='monte_carlo', color='tab:red', linewidth=4)
    plt.plot(state_map_obs.detach().cpu().numpy()[1, 1, 0], label='MAP', linewidth=4)
    plt.plot(state_obs_true.detach().cpu().numpy()[1, 1, 0], label='True', linewidth=4)
    plt.grid()
    plt.title('Displacement observations')

    plt.savefig('inverse_monte_carlo.pdf')
    
    plt.close()