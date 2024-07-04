
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

from data_assimilation_with_generative_ML.neural_network_models import DiT


class ParsDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path

        self.ids_list = range(1, 1001)

        self.min = 1e8
        self.max = 0
        
        self.perm_mean = 3.5748687267303465
        self.perm_std = 4.6395333366394045
        self.por_mean = 0.09433708190917969
        self.por_std = 0.03279830865561962


    def __len__(self):
        return 1000

    def __getitem__(self, idx):

        data = xr.load_dataset(f'{self.path}_{self.ids_list[idx]}.nc')

        pars = np.stack((data['Perm'].data, data['Por'].data), axis=0)

        pars = torch.tensor(pars, dtype=torch.float32)

        pars[0] = (pars[0] - self.perm_mean) / self.perm_std
        pars[1] = (pars[1] - self.por_mean) / self.por_std

        return pars

def main():
    #@title Set up the SDE

    device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

  
    sigma =  25.0#@param {'type':'number'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


    # score_model  = DiTScoreNet(marginal_prob_std=marginal_prob_std_fn, imsize=64)
    # score_model = torch.nn.DataParallel(ConvScoreNet(marginal_prob_std=marginal_prob_std_fn))
    # score_model = ConvScoreNet(marginal_prob_std=marginal_prob_std_fn, imsize=64)

    score_model = DiTScoreNet(marginal_prob_std=marginal_prob_std_fn, imsize=64, in_channels=2)
    # score_model.load_state_dict(torch.load('ckpt.pth'))
    score_model = score_model.to(device)

    

    n_epochs =   5000#@param {'type':'integer'}
    ## size of a mini-batch
    batch_size =  16 #@param {'type':'integer'}
    ## learning rate
    lr=5e-4 #@param {'type':'number'}

    # dataset = MNIST('.', train=True, transform=transform, download=True)
    
    # Sample only 5000 samples for training
    # dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), 5000))
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # x = xr.load_dataset(f'data/results64/simulation_results_realization_64x64_1.nc')
    dataset = ParsDataset(path='data/results64/simulation_results_realization_64x64')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    # dataset = dataset['facies code'].data[:]
    # dataset = torch.tensor(dataset, dtype=torch.float32)
    # num_samples = dataset.shape[0]

    # compute mean over the full dataset by looping over the dataloader
    # mean = 0.
    # num_items = 0
    # for x in data_loader:
    #     mean += x.mean().item() * x.shape[0]
    #     num_items += x.shape[0]
    # mean /= num_items
    
    # # compute standard deviation over the full dataset by looping over the dataloader
    # std = 0.
    # num_items = 0
    # for x in data_loader:
    #     std += x.std().item() * x.shape[0]
    #     num_items += x.shape[0]
    # std /= num_items

    # print(f'Mean: {mean}, Std: {std}')



    optimizer = Adam(score_model.parameters(), lr=lr, weight_decay=1e-8)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    tqdm_epoch = tqdm.trange(n_epochs)
    for epoch in tqdm_epoch:

        # Shuffle the dataset
        # random_ids = np.random.permutation(num_samples)
        # dataset = dataset[random_ids]

        avg_loss = 0.
        num_items = 0
        for x in data_loader:
        # for batch_ids in range(0, num_samples, batch_size):
            # x = dataset[batch_ids:batch_ids+batch_size]
            x = x.to(device)    
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
           
        scheduler.step()

        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), 'ckpt.pth')


        if epoch % 100 == 0:
            score_model.eval()
            with torch.no_grad():
                samples = pc_sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn, device=device, batch_size=8)
            samples = samples.cpu().numpy()
            samples = samples[:, 0]
            plt.figure()
            for i in range(8):
                plt.subplot(4, 4, i+1)
                plt.imshow(samples[i])
                plt.axis('off')

            x = next(iter(data_loader)).cpu().numpy()
            for i in range(8):
                plt.subplot(4, 4, i+9)
                plt.imshow(x[i, 0])
                plt.axis('off')
            plt.savefig('pc_samples.png')
            plt.close()

            with torch.no_grad():
                samples = ode_sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn, device=device, num_steps=500, batch_size=8)
            samples = samples.detach().cpu().numpy()
            samples = samples[:, 0]
            plt.figure()
            for i in range(8):
                plt.subplot(4, 4, i+1)
                plt.imshow(samples[i])
                plt.axis('off')

            x = next(iter(data_loader)).cpu().numpy()
            for i in range(8):
                plt.subplot(4, 4, i+9)
                plt.imshow(x[i, 0])
                plt.axis('off')
            plt.savefig('ode_samples.png')
            plt.close()
            score_model.train()
    return 0


if __name__ == "__main__":
    main()