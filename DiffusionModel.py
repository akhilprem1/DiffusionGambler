# TODO: Fix the loss functions of all diffusion models to use different subkeys. Check Song's JAX implementation.
# TODO: Make the B matrices for GRF non-trainable using lax.stop_gradient(). See GRF function in Song's code.

import jax
import jax.numpy as jnp
import flax.linen as lnn
from jax import jit, random
from typing import Sequence
import optax
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial


# ------------------------------------------------------------------------
# A VP diffusion model used to learn the joint distribution of (X,Y).
class DiffusionJoint(lnn.Module):
  features: Sequence[int]
  mapping_size: int
  num_dimensions: int # Dimensionality of the data vectors. Time not included.
  # sigma: float
  beta_min : float
  beta_max : float
  x_embed : bool = True
  maxL_prefactor : bool = False
  grf_scale_x : float = 10.0
  grf_scale_s : float = 10.0

  @lnn.compact
  def __call__(self, x, s):
    B_x = self.grf_scale_x * self.param('B_x', lnn.initializers.normal(), (self.mapping_size, self.num_dimensions))
    B_s = self.grf_scale_s * self.param('B_s', lnn.initializers.normal(), (self.mapping_size, 1))
    # Stop gradients from flowing through B_x and B_t. [NEW in Redux] 
    B_x = jax.lax.stop_gradient(B_x)
    B_s = jax.lax.stop_gradient(B_s)
    B_x = B_x if self.x_embed else None

    embed = self.input_mapping(s[..., None], B_s) # Convert from [batch_size,] to [batch_size, 1]
    embed = lnn.Dense(embed.shape[-1])(embed) # embed.shape[-1] = 2 * (mapping_size)
    embed = lnn.sigmoid(embed)
    pos = self.input_mapping(x, B_x)
    pos = lnn.Dense(pos.shape[-1])(pos) # This definitely helps improve learned scores.
    pos = lnn.sigmoid(pos)
    h = pos

    for feat in self.features[:-1]:
        tau = lnn.Dense(feat)(embed)
        h = lnn.Dense(feat)(h)
        h += tau
        h = lnn.LayerNorm()(h)
        h = lnn.sigmoid(h)

    # No time embedded in the last step, following Song's code.
    h = lnn.Dense(self.features[-1])(h)

    # Normalize the output.
    # h = h / jnp.expand_dims(self.marginal_prob_std(t), -1) #
    return h

  # Fourier feature mapping
  def input_mapping(self, x, B):
    if B is None:
      return x
    else:
      x_proj = (2.*jnp.pi*x) @ B.T
      return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

  def beta_at(self, s):
    return self.beta_min + (self.beta_max - self.beta_min) * s
  
  def bplus(self, x, s):
    return - self.beta_at(s)[..., None] * x / 2

  def sigma_at(self, s):
    return jnp.sqrt(self.beta_at(s))

  # This is the square root of expintbeta.
  def mu(self, s):
    return jnp.exp(- 0.5 * self.beta_min * s - 0.25 * (self.beta_max - self.beta_min) * (s ** 2))

  def marginal_prob_std(self, s):
    return jnp.sqrt((1 - self.mu(s)**2))
  
  def grad_logp_eq(self, x, s):
    return - x

  # The entropy matching loss.
  def loss(self, params, x, key, eps=1e-5, num_steps=1):
    x = jnp.tile(x[:, jnp.newaxis, :], (1, num_steps, 1)) # shape is [batch_size, num_steps, 2]
    key, subkey = random.split(key) # [NEW in Redux]
    random_s = random.uniform(subkey, x.shape[:-1]) * (1. - eps) + eps
    key, subkey = random.split(key)
    z = random.normal(subkey, x.shape)
    std = self.marginal_prob_std(random_s) # shape is [batch_size, num_steps]
    perturbed_x = x * self.mu(random_s)[..., None] + z * std[..., None] # Different for OU
    etheta = self.apply(params, perturbed_x, random_s)
    prefactor = (0.5 * (self.sigma_at(random_s)/std) ** 2) ** int(self.maxL_prefactor) # Maximum likelihood.
    return jnp.mean(prefactor * jnp.sum(((-perturbed_x + etheta) * std[..., None] + z) ** 2, axis=-1))


# ------------------------------------------------------------------------
# This is the VP process, with conditioning on the y variable.
# A diffusion model to learn the conditional distribution of X|Y.
# That is, only the x-component is diffused, and y is used for conditioning.
class DiffusionCond(lnn.Module):
  features : Sequence[int]
  mapping_size: int
  num_dimensions : int # D = D_X + D_Y
  beta_min : float
  beta_max : float
  D_X : int
  x_embed : bool = True
  maxL_prefactor : bool = False
  grf_scale_x : float = 10.0
  grf_scale_s : float = 10.0

  # NN accepts (D_X, D_Y, 1) dimensional inputs and produces D_X dimensional outputs.
  @lnn.compact
  def __call__(self, x, y, s):
    B_xy = self.grf_scale_x * self.param('B_xy', lnn.initializers.normal(), (self.mapping_size, self.num_dimensions))
    B_s = self.grf_scale_s * self.param('B_s', lnn.initializers.normal(), (self.mapping_size, 1))
    # Stop gradients from flowing through B_xy and B_s.
    B_xy = jax.lax.stop_gradient(B_xy)
    B_s = jax.lax.stop_gradient(B_s)
    B_xy = B_xy if self.x_embed else None

    embed = self.input_mapping(s[..., None], B_s) # Convert from [batch_size,] to [batch_size, 1]
    embed = lnn.Dense(embed.shape[-1])(embed) # embed.shape[-1] = 2 * (mapping_size)
    embed = lnn.sigmoid(embed)

    pos = jnp.concatenate([x, y], axis=-1)
    pos = self.input_mapping(pos, B_xy)
    pos = lnn.Dense(pos.shape[-1])(pos) # This definitely helps improve learned scores.
    pos = lnn.sigmoid(pos)
    h = pos

    for feat in self.features[:-1]:
        tau = lnn.Dense(feat)(embed)
        h = lnn.Dense(feat)(h)
        h += tau
        h = lnn.LayerNorm()(h)
        h = lnn.sigmoid(h)

    # No time embedded in the last step, following Song's code.
    h = lnn.Dense(self.D_X)(h)

    return h

  # Fourier feature mapping
  def input_mapping(self, x, B):
    if B is None:
      return x
    else:
      x_proj = (2.*jnp.pi*x) @ B.T
      return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

  # VP SDE
  def beta_at(self, s):
    return self.beta_min + (self.beta_max - self.beta_min) * s
  
  def bplus(self, x, s):
    return - self.beta_at(s)[..., None] * x / 2

  def sigma_at(self, s):
    return jnp.sqrt(self.beta_at(s))

  # This is the square root of expintbeta.
  def mu(self, s):
    return jnp.exp(- 0.5 * self.beta_min * s - 0.25 * (self.beta_max - self.beta_min) * (s ** 2))

  def marginal_prob_std(self, s):
    return jnp.sqrt((1 - self.mu(s)**2))
  
  def grad_logp_eq(self, x, s):
    return - x

  # The entropy matching loss.
  # We still accept the whole vectors r=(x,y) so we can use the existing training routines.
  def loss(self, params, r, key, eps=1e-5, num_steps=1):
    r = jnp.tile(r[:, jnp.newaxis, :], (1, num_steps, 1)) # shape is [batch_size, num_steps, D_X+D_Y]
    x, y = r[..., :self.D_X], r[..., self.D_X:]
    key, subkey = random.split(key) # [NEW in Redux]
    random_s = random.uniform(subkey, x.shape[:-1]) * (1. - eps) + eps
    key, subkey = random.split(key)
    z = random.normal(subkey, x.shape)
    std = self.marginal_prob_std(random_s) # shape is [batch_size, num_steps]
    perturbed_x = x * self.mu(random_s)[..., None] + z * std[..., None] # Different for OU
    etheta = self.apply(params, perturbed_x, y, random_s)
    prefactor = (0.5 * (self.sigma_at(random_s)/std) ** 2) ** int(self.maxL_prefactor) # Maximum likelihood.
    return jnp.mean(prefactor * jnp.sum(((-perturbed_x + etheta) * std[..., None] + z) ** 2, axis=-1))
  

# ------------------------------------------------------------------------
# A diffusion model class that just stores the diffusion parameters and time scaling.
class DiffusionBare():
  def __init__(self, beta_min, beta_max):

    self.beta_min = beta_min
    self.beta_max = beta_max

  # VP SDE
  def beta_at(self, t):
    return self.beta_min + (self.beta_max - self.beta_min) * t
  
  def bplus(self, x, t):
    return - self.beta_at(t)[..., None] * x / 2

  def sigma_at(self, t):
    return jnp.sqrt(self.beta_at(t))

  # This is the square root of expintbeta.
  def mu(self, t):
    return jnp.exp(- 0.5 * self.beta_min * t - 0.25 * (self.beta_max - self.beta_min) * (t ** 2))

  def marginal_prob_std(self, t):
    return jnp.sqrt((1 - self.mu(t)**2))

  def grad_logp_eq(self, x, t):
    return - x
  

# ------------------------------------------------------------------------
# A class to make a NumPy array into a TF dataset.
# Works for any dimensions. The name is a legacy artifact.
class Data2D(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data) # same as self.data.shape[0]

  def __getitem__(self, idx):
    return self.data[idx] # same as self.data[idx,]