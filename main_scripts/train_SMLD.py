
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

from data_assimilation_with_generative_ML.datasets import ParsDataset
from data_assimilation_with_generative_ML.diffusion_models import (
    ConvScoreNet,
    DiTScoreNet,
    loss_fn,
    marginal_prob_std,
    diffusion_coeff,
    pc_sampler,
    ode_sampler,
)

from data_assimilation_with_generative_ML.neural_network_models import DiT
from data_assimilation_with_generative_ML.transformer_layers import UNet_Tranformer

def main():

    device = 'cuda'


    sigma =  25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    score_model = UNet_Tranformer(
        marginal_prob_std=marginal_prob_std_fn, 
        in_channels=1,
        channels=[16, 32, 64, 128], # [4, 8, 16, 32]
        imsize=64,
    )
    # score_model.load_state_dict(torch.load('diffusion_model.pth'))
    score_model = score_model.to(device)

    n_epochs = 5000
    batch_size = 16
    lr=5e-4

    dataset = ParsDataset(path='data/geodata/processed_DARTS_simulation_realization')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = Adam(score_model.parameters(), lr=lr, weight_decay=1e-8)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    tqdm_epoch = tqdm.trange(n_epochs)
    for epoch in tqdm_epoch:

        avg_loss = 0.
        num_items = 0
        for x in data_loader:
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
        torch.save(score_model.state_dict(), 'diffusion_model.pth')


        if epoch % 50 == 0:
            x = next(iter(data_loader)).cpu().numpy()

            score_model.eval()
            with torch.no_grad():
                samples = pc_sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn, device=device, batch_size=8, num_steps=10)
            samples = samples.cpu().numpy()
            samples = samples[:, 0]
            plt.figure()
            for i in range(8):
                plt.subplot(4, 4, i+1)
                plt.imshow(samples[i], vmin=x.min(), vmax=x.max())
                plt.colorbar()
                plt.axis('off')

            for i in range(8):
                plt.subplot(4, 4, i+9)
                plt.imshow(x[i, 0], vmin=x.min(), vmax=x.max())
                plt.colorbar()
                plt.axis('off')
            plt.savefig('pc_samples.pdf')
            plt.close()

            with torch.no_grad():
                samples = ode_sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn, device=device, num_steps=10, batch_size=8)
            samples = samples.detach().cpu().numpy()
            samples = samples[:, 0]
            plt.figure()
            for i in range(8):
                plt.subplot(4, 4, i+1)
                plt.imshow(samples[i], vmin=x.min(), vmax=x.max())
                plt.colorbar()
                plt.axis('off')

            for i in range(8):
                plt.subplot(4, 4, i+9)
                plt.imshow(x[i, 0], vmin=x.min(), vmax=x.max())
                plt.colorbar()
                plt.axis('off')
            plt.savefig('ode_samples.pdf')
            plt.close()
            score_model.train()
    return 0


if __name__ == "__main__":
    main()



# dataset = dataset['facies code'].data[:]
# dataset = torch.tensor(dataset, dtype=torch.float32)
# num_samples = dataset.shape[0]

# compute mean over the full dataset by looping over the dataloader
# mean = 0.
# num_items = 0

# min = 1e8
# for x in data_loader:
#     mean += x.mean().item()# * x.shape[0]
#     num_items += 1#x.shape[0]

#     if x.min() < min:
#         min = x.min()
# mean /= num_items

# # compute standard deviation over the full dataset by looping over the dataloader
# std = 0.
# num_items = 0
# max = -1e8
# for x in data_loader:
#     std += x.std().item() * x.shape[0]
#     num_items += x.shape[0]

#     if x.max() > max:
#         max = x.max()

# std /= num_items

# print(f'Mean: {mean}, Std: {std}')
# print(f'Min: {min}, Max: {max}')
