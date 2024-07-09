
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

class ForwardModelDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path

        self.ids_list = range(0, 1342)

        self.perm_mean = -1.3754382634162903 #3.5748687267303465
        self.perm_std = 3.6160271644592283 #4.6395333366394045
        # self.por_mean = 0.09433708190917969
        # self.por_std = 0.03279830865561962


        self.pressure_mean = 59.839262017194635 # 336.26219848632815
        self.pressure_std = 16.946480998339133 #130.7361669921875
        self.pressure_min = 51.5
        self.pressure_max = 299.43377685546875
        self.co2_mean = 0.9997552563110158 #0.03664950348436832
        self.co2_std = 0.004887599662507033 #0.13080736815929414

        self.ft_min = 6.760696180663217e-08 #0.0
        self.ft_max = 0.0009333082125522196 #28074.49609375



    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):

        data = xr.load_dataset(f'{self.path}_{self.ids_list[idx]}.nc')

        state = np.concat((data['PRESSURE'].data, data['H2O'].data), axis=1)
        
        pars = data['Perm'].data[0] # np.stack((data['Perm'].data, data['Por'].data), axis=0)
        ft = data['c_0_rate'].data[0]


        state = torch.tensor(state, dtype=torch.float32)
        pars = torch.tensor(pars, dtype=torch.float32)
        pars = torch.log(pars)
        ft = torch.tensor(ft, dtype=torch.float32)

        state = torch.permute(state, (1, 2, 3, 0))

        # state[0] = (state[0] - self.pressure_mean) / self.pressure_std

        state[0] = (state[0] - self.pressure_min) / (self.pressure_max - self.pressure_min)
        #state[1] = (state[1] - self.co2_mean) / self.co2_std
        pars[0] = (pars[0] - self.perm_mean) / self.perm_std
        #pars[1] = (pars[1] - self.por_mean) / self.por_std

        ft = (ft - self.ft_min) / (self.ft_max - self.ft_min)


        return state, pars, ft


def main():
    #@title Set up the SDE

    device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

    model_args = {
        'img_size': 64,
        'dim': 256,
        'patch_size': 8,
        'depth': 4,
        'heads': 8,
        'mlp_dim': 1024,
        'k': 128,
        'in_channels': 3,
    }
    forward_model = ForwardModel(model_args)
    # forward_model.load_state_dict(torch.load('forward_model.pth'))

    forward_model = forward_model.to(device)


    n_epochs =   5000#@param {'type':'integer'}
    ## size of a mini-batch
    batch_size = 4 #@param {'type':'integer'}
    ## learning rate
    lr=5e-4 #@param {'type':'number'}

    # data = xr.load_dataset('data/geodata/processed_DARTS_simulation_realization_0.nc')
    # state = np.concat((data['PRESSURE'].data, data['H2O'].data), axis=1)
    # state = torch.permute(state, (1, 2, 3, 0))
    
    # dataset = ForwardModelDataset(path='data/results64/simulation_results_realization_64x64')
    dataset = ForwardModelDataset(path='data/geodata/processed_DARTS_simulation_realization')
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = Adam(forward_model.parameters(), lr=lr, weight_decay=1e-8)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    loss_fn = nn.MSELoss()
    tqdm_epoch = tqdm.trange(n_epochs)
    num_steps = 1
    for epoch in tqdm_epoch:

        avg_loss = 0.
        num_items = 0

        if epoch % 250 == 0 and num_steps < 10:
            num_steps += 1

        for state, pars, ft in data_loader:


            pars = pars.to(device)
            ft = ft.to(device)


            i = np.random.randint(0, ft.shape[-1]-num_steps)

            batch_state_0 = state[:, :, :, :, i].to(device)
            batch_state_1 = state[:, :, :, :, (i+1):(i+1+num_steps)].to(device)

            pred_state_1 = forward_model(batch_state_0, pars, ft[:, i:i+num_steps])

            loss = loss_fn(pred_state_1, batch_state_1)

            
            # if epoch < 250:

            #     i = np.random.randint(0, ft.shape[-1]-1)

            #     batch_state_0 = state[:, :, :, :, i].to(device)
            #     batch_state_1 = state[:, :, :, :, i+1].to(device)

            #     pred_state_1 = forward_model.compute_one_step(batch_state_0, pars, ft[:, i])

            #     loss = loss_fn(pred_state_1, batch_state_1)
                
            # elif epoch >= 250 and epoch < 500:

            #     i = np.random.randint(0, ft.shape[-1]-11)

            #     batch_state_0 = state[:, :, :, :, i].to(device)
            #     batch_state_1 = state[:, :, :, :, (i+1):(i+11)].to(device)

            #     pred_state_1 = forward_model(batch_state_0, pars, ft[:, i:i+10])

            #     loss = loss_fn(pred_state_1, batch_state_1)

            # else:
                
            #     state = state.to(device)
            #     state_0 = state[:, :, :, :, 0].to(device)

            #     pred_state = forward_model(state_0, pars, ft)
            #     loss = loss_fn(pred_state, state)
                    

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


        if epoch % 5 == 0:

            forward_model.eval()

            state, pars, ft = next(iter(data_loader))
            state = state.to(device)
            pars = pars.to(device)
            ft = ft.to(device)

            state_0 = state[:, :, :, :, 0]

            pred_state = forward_model(state_0, pars, ft)

            mse = loss_fn(pred_state, state)
            print(f'Val MSE: {mse.item()}')
            
            pred_state = pred_state.detach().cpu().numpy()


            x = state.cpu().numpy()
            t_vec = [15, 30, 45, 60]
            pres_min = x[0, 0].min()
            pres_max = x[0, 0].max()
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
                plt.subplot(3, 4, i+9)
                plt.imshow(x[0, 0, :, :, t_vec[i]]-pred_state[0, 0, :, :, t_vec[i]], vmin=-pres_min, vmax=pres_max)
                plt.colorbar()
                plt.axis('off')


            plt.savefig('forward_model_samples_pressure.png')
            plt.close()

            co2_min = x[0, 1].min()
            co2_max = x[0, 1].max()

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
                plt.subplot(3, 4, i+9)
                plt.imshow(x[0, 1, :, :, t_vec[i]]-pred_state[0, 1, :, :, t_vec[i]])
                plt.colorbar()
                plt.axis('off')

            plt.savefig('forward_model_samples_co2.png')
            plt.close()



            forward_model.train()


    return 0


if __name__ == "__main__":
    main()