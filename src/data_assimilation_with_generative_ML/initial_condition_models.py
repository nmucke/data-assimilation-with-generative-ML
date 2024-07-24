
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
        return x

    def forward(self, pars, ft):

        x = self.dit(x, ft)
        x = self.final_layer(x)

        return x