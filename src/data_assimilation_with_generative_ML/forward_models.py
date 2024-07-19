
import pdb
import torch
import torch.nn as nn
import math

from data_assimilation_with_generative_ML.neural_network_models import DiT
from data_assimilation_with_generative_ML.transformer_layers import UNet_Tranformer


class ForwardModel(nn.Module):
    def __init__(
        self, 
        model_args
    ):
        super().__init__()

        # self.dit = DiT(**model_args)
        self.dit = UNet_Tranformer(**model_args)

        self.final_layer = nn.Conv2d(
            in_channels=model_args['in_channels'], 
            out_channels=3, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            bias=False
        )

    def compute_one_step(self, x, pars, ft):
        x = torch.cat([x, pars], dim=1)
        x = self.dit(x, ft)
        x = self.final_layer(x)
        return x

    def forward(self, x0, pars, ft):

        x_out = x0.unsqueeze(-1)

        for i in range(ft.shape[-1]-1):
            x_next = self.compute_one_step(x_out[:, :, :, :, -1], pars, ft[:, i])

            x_out = torch.cat([x_out, x_next.unsqueeze(-1)], dim=-1)

        return x_out