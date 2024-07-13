

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

from data_assimilation_with_generative_ML.neural_network_models import DiT


class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class Usample(nn.Module):
    """Upsampling layer for the U-Net architecture."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))

class ConvScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256, imsize=28, in_channels=1):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    self.imsize = imsize
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(in_channels, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    # self.tconv4 = Usample(channels[3], channels[2])#nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False, output_padding=1)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    # self.tconv3 = Usample(channels[2] + channels[2], channels[1]) #nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)   
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    # self.tconv2 = Usample(channels[1] + channels[1], channels[0]) #nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)   
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    # self.tconv1 = Usample(channels[0] + channels[0], 1) #nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))  
    # Encoding path
    h1 = self.conv1(x)    
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h

class DiTScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, imsize=28, in_channels=1):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()

    self.imsize = imsize
    self.in_channels = in_channels

    self.DiT = DiT(imsize, dim=256, patch_size=8, depth=4, heads=4, mlp_dim=512, k=128, in_channels=in_channels)

    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t): 
    
    h = self.DiT(x, t)

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h

def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of p_{0t}(x(t) | x(0)).

    Args:    
        t: A vector of time steps.
        sigma: The sigma in our SDE.  

    Returns:
        The standard deviation.
    """    
    #   t = torch.tensor(t, device=t.device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The sigma in our SDE.

    Returns:
        The vector of diffusion coefficients.
    """

    return sigma**t


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss


signal_to_noise_ratio = 0.16 #@param {'type':'number'}

## The number of sampling steps.
def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64, 
               num_steps=500, 
               snr=signal_to_noise_ratio,                
               device='cuda',
               eps=1e-3):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation
            of the perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient 
            of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps. 
            Equivalent to the number of discretized time steps.    
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.

    Returns: 
        Samples.
    """

    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, score_model.in_channels, score_model.imsize, score_model.imsize, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

        # The last step does not include any noise
    return x_mean


class ODE_wrapper(nn.Module):

    def __init__(self, score_model, shape, device):
        super(ODE_wrapper, self).__init__()

        self.score_model = score_model
        self.shape = shape
        self.device = device
        self.nfev = 0
 
    
    def forward(self, t, x):
        """The ODE function for use by the ODE solver."""
        self.nfev += 1

        time_steps = torch.ones((self.shape[0],), device=self.device) * t    

        g = diffusion_coeff(t, sigma=25.0)
        return  -0.5 * (g**2) * self.score_model(x, time_steps)

## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5 #@param {'type': 'number'}
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                atol=error_tolerance, 
                rtol=error_tolerance, 
                device='cuda', 
                z=None,
                num_steps=500,
                eps=1e-3):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that returns the standard deviation 
        of the perturbation kernel.
        diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        atol: Tolerance of absolute errors.
        rtol: Tolerance of relative errors.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        z: The latent code that governs the final sample. If None, we start from p_1;
        otherwise, we start from the given z.
        eps: The smallest time step for numerical stability.
    """
    if z is not None:
       batch_size = z.shape[0]
       
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if z is None:
        init_x = torch.randn(batch_size, score_model.in_channels, score_model.imsize, score_model.imsize, device=device) \
        * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z * marginal_prob_std(t)[:, None, None, None]
        
    shape = init_x.shape
    device = init_x.device

    # def score_eval_wrapper(sample, time_steps):
    #     """A wrapper of the score-based model for use by the ODE solver."""
    #     sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    #     time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))      
    #     score = score_model(sample, time_steps)
    #     return score.reshape((-1,))
        #return score.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    # def ode_func(t, x):        
    #     """The ODE function for use by the ODE solver."""
    #     time_steps = np.ones((shape[0],)) * t    
    #     g = diffusion_coeff(torch.tensor(t))
    #     return  -0.5 * (g**2) * score_model(x, time_steps)#score_eval_wrapper(x, time_steps)
    
    # Wrap the ODE function for use by the ODE solver.
    ode_func = ODE_wrapper(score_model, shape, device)

    t_vec = torch.linspace(1., eps, num_steps, device=device)
    
    res = odeint_adjoint(ode_func, init_x, t_vec, method='rk4', rtol=rtol, atol=atol)
    
    # Run the black-box ODE solver.
    # res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
    # print(f"Number of function evaluations: {res.nfev}")
    # x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return res[-1]

