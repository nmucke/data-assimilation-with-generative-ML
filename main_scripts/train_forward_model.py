
import functools
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
import matplotlib.pyplot as plt
import xarray as xr
from torchdiffeq import odeint_adjoint

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
from data_assimilation_with_generative_ML.neural_network_models import DiT



def main():
    
    device = 'cuda'

    # model_args = {
    #     'img_size': 64,
    #     'dim': 256,
    #     'patch_size': 8,
    #     'depth': 4,
    #     'heads': 8,
    #     'mlp_dim': 1024,
    #     'k': 128,
    #     'in_channels': 4,
    # }
    model_args = {
        'marginal_prob_std':None, 
        'in_channels':4,
        'channels':[8, 16, 32, 64],
        'imsize':64,
    }
    forward_model = ForwardModel(model_args)
    forward_model.load_state_dict(torch.load('forward_model.pth'))

    forward_model = forward_model.to(device)

    n_epochs =   5000
    batch_size = 8
    lr=5e-4 
    
    path = 'data/geodata/processed_DARTS_simulation_realization'
    dataset = ForwardModelDataset(path=path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = Adam(forward_model.parameters(), lr=lr, weight_decay=1e-8)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    loss_fn = nn.MSELoss()
    tqdm_epoch = tqdm.trange(n_epochs)
    num_steps = 10
    for epoch in tqdm_epoch:

        avg_loss = 0.
        num_items = 0

        # if epoch % 100 == 0 and num_steps < 10:
        #     num_steps += 1

        for state, pars, ft in data_loader:

            pars = pars.to(device)
            ft = ft.to(device)

            pars_noise = torch.randn_like(pars) * 1e-3
            pars = pars + pars_noise

            i = np.random.randint(0, ft.shape[-1]-num_steps)

            batch_state_0 = state[:, :, :, :, i].to(device)

            batch_state_1 = state[:, :, :, :, (i+1):(i+1+num_steps)].to(device)

            pred_state_1 = forward_model(batch_state_0, pars, ft[:, i:i+num_steps])

            loss = loss_fn(pred_state_1, batch_state_1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            num_items += 1

        scheduler.step()

        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

        # Update the checkpoint after each epoch of training.
        torch.save(forward_model.state_dict(), 'forward_model.pth')


        if epoch % 50 == 0:

            forward_model.eval()

            state, pars, ft = next(iter(data_loader))
            state = state.to(device)
            pars = pars.to(device)
            ft = ft.to(device)

            state_0 = state[:, :, :, :, 0]

            pred_state = forward_model(state_0, pars, ft)

            # Compute RMSE
            RMSE_pres = torch.sqrt(loss_fn(pred_state[:, 0], state[:, 0]))
            print(f'RMSE pressure: {RMSE_pres.item()}')

            RMSE_co2 = torch.sqrt(loss_fn(pred_state[:, 1], state[:, 1]))
            print(f'RMSE co2: {RMSE_co2.item()}')

            RMSE_U_z = torch.sqrt(loss_fn(pred_state[:, 2], state[:, 2]))
            print(f'RMSE U_z: {RMSE_U_z.item()}')

            pred_state = pred_state.detach().cpu().numpy()

            x = state.cpu().numpy()
            t_vec = [30, 60, 90, 119]
            pres_min = x[0, 0].min()
            pres_max = x[0, 0].max()

            err_min = (x[0, 0] - pred_state[0, 0]).min()
            err_max = (x[0, 0] - pred_state[0, 0]).max()
            plt.figure()
            for i in range(4):
                plt.subplot(3, 4, i+1)
                plt.imshow(pred_state[0, 0, :, :, t_vec[i]], vmin=-pres_min, vmax=pres_max)
                plt.colorbar()
                plt.axis('off')

            for i in range(4):
                plt.subplot(3, 4, i+5)
                plt.imshow(x[0, 0, :, :, t_vec[i]], vmin=-pres_min, vmax=pres_max)
                plt.colorbar()
                plt.axis('off')

            for i in range(4):
                RMSE = torch.sqrt(loss_fn(torch.tensor(x[0, 0, :, :, t_vec[i]]), torch.tensor(pred_state[0, 0, :, :, t_vec[i]])))
                plt.subplot(3, 4, i+9)
                plt.imshow(x[0, 0, :, :, t_vec[i]]-pred_state[0, 0, :, :, t_vec[i]], vmin=-err_min, vmax=err_max)
                plt.title(f't={t_vec[i]:0.0f},RMSE={RMSE:0.3f}', fontsize=8)
                plt.colorbar()
                plt.axis('off')


            plt.savefig('forward_model_samples_pressure.pdf')
            plt.close()

            err_min = (x[0, 1] - pred_state[0, 1]).min()
            err_max = (x[0, 1] - pred_state[0, 1]).max()

            plt.figure()
            for i in range(4):
                plt.subplot(3, 4, i+1)
                plt.imshow(pred_state[0, 1, :, :, t_vec[i]], vmin=0, vmax=1)
                plt.colorbar()
                plt.axis('off')

            for i in range(4):
                plt.subplot(3, 4, i+5)
                plt.imshow(x[0, 1, :, :, t_vec[i]], vmin=0, vmax=1)
                plt.colorbar()
                plt.axis('off')

            for i in range(4):
                RMSE = torch.sqrt(loss_fn(torch.tensor(x[0, 1, :, :, t_vec[i]]), torch.tensor(pred_state[0, 1, :, :, t_vec[i]])))
                plt.subplot(3, 4, i+9)
                plt.imshow(x[0, 1, :, :, t_vec[i]]-pred_state[0, 1, :, :, t_vec[i]], vmin=-err_min, vmax=err_max)
                plt.title(f't={t_vec[i]:0.0f}, RMSE={RMSE:0.3f}', fontsize=8)
                plt.colorbar()
                plt.axis('off')

            plt.savefig('forward_model_samples_co2.pdf')
            plt.close()

            err_min = (x[0, 2] - pred_state[0, 2]).min()
            err_max = (x[0, 2] - pred_state[0, 2]).max()
            U_z_min = x[0, 2].min()
            U_z_max = x[0, 2].max()

            plt.figure()
            for i in range(4):
                plt.subplot(3, 4, i+1)
                plt.imshow(pred_state[0, 2, :, :, t_vec[i]], vmin=U_z_min, vmax=U_z_max)
                plt.colorbar()
                plt.axis('off')

            for i in range(4):
                plt.subplot(3, 4, i+5)
                plt.imshow(x[0, 2, :, :, t_vec[i]], vmin=U_z_min, vmax=U_z_max)
                plt.colorbar()
                plt.axis('off')

            for i in range(4):
                RMSE = torch.sqrt(loss_fn(torch.tensor(x[0, 2, :, :, t_vec[i]]), torch.tensor(pred_state[0, 2, :, :, t_vec[i]])))
                plt.subplot(3, 4, i+9)
                plt.imshow(x[0, 2, :, :, t_vec[i]]-pred_state[0, 2, :, :, t_vec[i]], vmin=-err_min, vmax=err_max)
                plt.title(f't={t_vec[i]:0.0f}, RMSE={RMSE:0.3f}', fontsize=8)
                plt.colorbar()
                plt.axis('off')

            plt.savefig('forward_model_samples_Uz.pdf')
            plt.close()



            forward_model.train()

    return 0


if __name__ == "__main__":
    main()