import jax
import jax.numpy as jnp
import flax.linen as lnn
from jax import jit, random
from typing import Sequence, Tuple
import numpy as np
from scipy.integrate import solve_ivp


#------------------------------------------------------------------------
# The neural network and its components. From NeuralEntropyRedux.ipynb.

class GaussianFourierProjection(lnn.Module):
  """Gaussian random features for encoding time steps."""  
  embed_dim: int
  scale: float = 30.
  @lnn.compact
  def __call__(self, x): 
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    W = self.param('W', jax.nn.initializers.normal(stddev=self.scale), 
                 (self.embed_dim // 2, ))
    W = jax.lax.stop_gradient(W)
    x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Dense(lnn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""  
  output_dim: int  
  
  @lnn.compact
  def __call__(self, x):
    return lnn.Dense(self.output_dim)(x)[:, None, None, :]


# The unconditional U-net and the associated parameters. Works for images with shape (28, 28, 1).
class NeuralCoreUnet(lnn.Module):
  channels: Tuple[int] = (32, 64, 128, 256)
  mapping_size: int = 256
  grf_scale_t : float = 30.0
  img_shape : tuple = (28, 28, 1)

  @lnn.compact
  def __call__(self, x, s): 
    # The swish activation function
    act = lnn.swish

    # Obtain the Gaussian random feature embedding for t   
    embed = act(lnn.Dense(self.mapping_size)(
        GaussianFourierProjection(embed_dim=self.mapping_size, scale=self.grf_scale_t)(s)))
        
    # Encoding path
    # 'VALID' adds no padding. Same as leaving out the padding argument completely.
    h1 = lnn.Conv(self.channels[0], (3, 3), (1, 1), padding='VALID', use_bias=False)(x)   
    ## Incorporate information from t
    h1 += Dense(self.channels[0])(embed)
    ## Group normalization
    h1 = lnn.GroupNorm(4)(h1)    
    h1 = act(h1)
    h2 = lnn.Conv(self.channels[1], (3, 3), (2, 2), padding='VALID', use_bias=False)(h1)
    h2 += Dense(self.channels[1])(embed)
    h2 = lnn.GroupNorm()(h2)        
    h2 = act(h2)
    h3 = lnn.Conv(self.channels[2], (3, 3), (2, 2), padding='VALID', use_bias=False)(h2)
    h3 += Dense(self.channels[2])(embed)
    h3 = lnn.GroupNorm()(h3)
    h3 = act(h3)
    h4 = lnn.Conv(self.channels[3], (3, 3), (2, 2), padding='VALID', use_bias=False)(h3)
    h4 += Dense(self.channels[3])(embed)
    h4 = lnn.GroupNorm()(h4)    
    h4 = act(h4)

    # Decoding path
    h = lnn.Conv(self.channels[2], (3, 3), (1, 1), padding=((2, 2), (2, 2)),
                  input_dilation=(2, 2), use_bias=False)(h4)    
    ## Skip connection from the encoding path
    h += Dense(self.channels[2])(embed)
    h = lnn.GroupNorm()(h)
    h = act(h)
    h = lnn.Conv(self.channels[1], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
                  input_dilation=(2, 2), use_bias=False)(
                      jnp.concatenate([h, h3], axis=-1)
                  )
    h += Dense(self.channels[1])(embed)
    h = lnn.GroupNorm()(h)
    h = act(h)
    h = lnn.Conv(self.channels[0], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
                  input_dilation=(2, 2), use_bias=False)(
                      jnp.concatenate([h, h2], axis=-1)
                  )    
    h += Dense(self.channels[0])(embed)    
    h = lnn.GroupNorm()(h)
    h = act(h)
    h = lnn.Conv(1, (3, 3), (1, 1), padding=((2, 2), (2, 2)))(
        jnp.concatenate([h, h1], axis=-1)
    )

    return h

#------------------------------------------------------------------------
# Base class for the unconditional diffusion models.
class DiffusionImages():
  def __init__(self, neural_core, maxL_prefactor=False, eps0=1e-5, epsT=0):
    self.neural_core = neural_core
    self.maxL_prefactor = maxL_prefactor

    self.eps0 = eps0
    self.epsT = epsT

  def init(self, key, x, s):
    D = jnp.prod(jnp.array(x.shape[-3:]))
    self.S0 = 0.5 * D * jnp.log(2 * jnp.pi * jnp.e * self.marginal_prob_std(1)**2)
    
    return self.neural_core.init(key, x, s)
  
  def apply(self, params, x, s):
    return self.neural_core.apply(params, x, s)

  # The entropy matching loss.
  def loss(self, params, x, key, num_steps=1):
    x = jnp.tile(x[:, jnp.newaxis, :, :, :], (1, num_steps, 1, 1, 1)) # shape is [batch_size, num_steps, 2]
    x = x.reshape(-1, *x.shape[-3:]) # Flatten all but last 3 dimensions, so shape is [batch_size * num_steps w, h, channels]

    key, subkey = random.split(key)
    random_s = random.uniform(subkey, (*x.shape[:-3],), minval=self.eps0, maxval=1.-self.epsT) # shape is [batch_size * num_steps]
    key, subkey = random.split(key)
    z = random.normal(subkey, x.shape)
    std = self.marginal_prob_std(random_s)

    perturbed_x = x * self.mu(random_s)[..., None, None, None] + z * std[..., None, None, None] # Different for OU
    etheta = self.apply(params, perturbed_x, random_s)
    prefactor = (0.5 * (self.sigma_at(random_s)/std) ** 2) ** int(self.maxL_prefactor) # Maximum likelihood.
    return jnp.mean(prefactor * jnp.sum(((self.grad_logp_eq(perturbed_x, random_s) + etheta) * std[..., None, None, None] + z) ** 2, axis=(1,2,3)))


# We inherit from the above base class to implement the different diffusion processses.
class DiffusionImagesEM(DiffusionImages):
  def __init__(self, neural_core, beta_min, beta_max, kappa, maxL_prefactor=False):
    """
    Initializes the diffusion model with a provided neural network module
    and the diffusion process parameters.
    """
    super().__init__(neural_core, maxL_prefactor)
    self.beta_min = beta_min
    self.beta_max = beta_max
    self.kappa = kappa
  
  # VP SDE with kappa
  def beta_at(self, s):
    return self.beta_min + (self.beta_max - self.beta_min) * s
  
  def bplus(self, x, s):
    return - self.beta_at(s)[..., None, None, None] * x / 2

  def sigma_at(self, s):
    return self.kappa * jnp.sqrt(self.beta_at(s))

  # This is the square root of expintbeta.
  def mu(self, s):
    return jnp.exp(- 0.5 * self.beta_min * s - 0.25 * (self.beta_max - self.beta_min) * (s ** 2))

  def marginal_prob_std(self, s):
    return self.kappa * jnp.sqrt((1 - self.mu(s)**2))
  
  def grad_logp_eq(self, x, s):
    return - x / self.kappa**2


class DiffusionImagesSL(DiffusionImages):
  def __init__(self, neural_core, Sigma_0, maxL_prefactor=False):
    """
    Initializes the diffusion model with a provided neural network module
    and the diffusion process parameters.
    """
    # Stop shy of T=1 for SLDM.
    super().__init__(neural_core, maxL_prefactor, eps0=1e-5, epsT=1e-5)
    self.Sigma_0 = Sigma_0

  def bplus(self, x, s):
    return - x / (1 - s[..., None, None, None])

  def sigma_at(self, s):
    return self.Sigma_0 * jnp.sqrt(2/(1 - s))
  
  def mu(self, s):
    return 1-s

  def marginal_prob_std(self, s):
    return self.Sigma_0 * jnp.sqrt(1 - (1-s)**2)
  
  def grad_logp_eq(self, x, s):
    return - x / self.Sigma_0**2
  

#------------------------------------------------------------------------
# This class helps us use train_diffusion_with_checkpoints
# without using 'for batch, _ in data_loader:'.

class ImageOnlyDataset:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image
    

#------------------------------------------------------------------------
# The conditional U-net for MNIST from NeuralEntropyRedux.
class NeuralCoreUnetConditional(lnn.Module):
  channels: Tuple[int] = (32, 64, 128, 256)
  mapping_size: int = 256
  grf_scale_t : float = 30.0
  num_classes : int = 10  # Number of classes
  img_shape : tuple = (28, 28, 1)

  @lnn.compact
  def __call__(self, x, c, t):
    # The swish activation function
    act = lnn.swish

    # Mix time and class embeddings into one.
    # Obtain the Gaussian random feature embedding for t   
    embed_t = act(lnn.Dense(self.mapping_size)(
        GaussianFourierProjection(embed_dim=self.mapping_size, scale=self.grf_scale_t)(t)))
    
    # Create the class embedding also with the required mapping size
    class_embedding = lnn.Embed(self.num_classes, self.mapping_size)(c)

    # Combine timestep and class embeddings
    embed = embed_t + class_embedding
       
    # Encoding path
    # 'VALID' adds no padding. Same as leaving out the padding argument completely.
    h1 = lnn.Conv(self.channels[0], (3, 3), (1, 1), padding='VALID', use_bias=False)(x)   
    ## Incorporate information from t
    h1 += Dense(self.channels[0])(embed)
    ## Group normalization
    h1 = lnn.GroupNorm(4)(h1)    
    h1 = act(h1)
    h2 = lnn.Conv(self.channels[1], (3, 3), (2, 2), padding='VALID', use_bias=False)(h1)
    h2 += Dense(self.channels[1])(embed)
    h2 = lnn.GroupNorm()(h2)        
    h2 = act(h2)
    h3 = lnn.Conv(self.channels[2], (3, 3), (2, 2), padding='VALID', use_bias=False)(h2)
    h3 += Dense(self.channels[2])(embed)
    h3 = lnn.GroupNorm()(h3)
    h3 = act(h3)
    h4 = lnn.Conv(self.channels[3], (3, 3), (2, 2), padding='VALID', use_bias=False)(h3)
    h4 += Dense(self.channels[3])(embed)
    h4 = lnn.GroupNorm()(h4)    
    h4 = act(h4)

    # Decoding path
    h = lnn.Conv(self.channels[2], (3, 3), (1, 1), padding=((2, 2), (2, 2)),
                  input_dilation=(2, 2), use_bias=False)(h4)    
    ## Skip connection from the encoding path
    h += Dense(self.channels[2])(embed)
    h = lnn.GroupNorm()(h)
    h = act(h)
    h = lnn.Conv(self.channels[1], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
                  input_dilation=(2, 2), use_bias=False)(
                      jnp.concatenate([h, h3], axis=-1)
                  )
    h += Dense(self.channels[1])(embed)
    h = lnn.GroupNorm()(h)
    h = act(h)
    h = lnn.Conv(self.channels[0], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
                  input_dilation=(2, 2), use_bias=False)(
                      jnp.concatenate([h, h2], axis=-1)
                  )    
    h += Dense(self.channels[0])(embed)
    h = lnn.GroupNorm()(h)
    h = act(h)
    h = lnn.Conv(1, (3, 3), (1, 1), padding=((2, 2), (2, 2)))(
        jnp.concatenate([h, h1], axis=-1)
    )

    return h

  
#------------------------------------------------------------------------
# Base class for generic OU process. See 'Elucidating the design space...' paper.
class DiffusionImagesConditional():
  def __init__(self, neural_core, maxL_prefactor=False, eps0=1e-5, epsT=0):
    self.neural_core = neural_core
    self.maxL_prefactor = maxL_prefactor

    self.eps0 = eps0
    self.epsT = epsT

  def init(self, key, x, c, s):
    D = jnp.prod(jnp.array(x.shape[-3:]))
    self.S0 = 0.5 * D * jnp.log(2 * jnp.pi * jnp.e * self.marginal_prob_std(1)**2)
    
    return self.neural_core.init(key, x, c, s)
  
  def apply(self, params, x, c, s):
    return self.neural_core.apply(params, x, c, s)
  
  # The entropy matching loss for conditional generation.
  def loss(self, params, x, c, key, num_steps=1):
    x = jnp.tile(x[:, jnp.newaxis, :, :, :], (1, num_steps, 1, 1, 1)) # shape is [batch_size, num_steps, w, h, channels]
    x = x.reshape(-1, *x.shape[-3:]) # Flatten all but last 3 dimensions, so shape is [batch_size * num_steps w, h, channels]
    c = jnp.repeat(c, num_steps) # Same as jnp.tile(c[:, jnp.newaxis], (1, num_steps)).reshape(-1)
    
    # Using a different random_s for each class. Density helper will tile.
    key, subkey = jax.random.split(key)
    random_s = random.uniform(subkey, (*x.shape[:-3],), minval=self.eps0, maxval=1.-self.epsT) # shape is [batch_size * num_steps]
    key, subkey = jax.random.split(key)
    z = random.normal(subkey, x.shape)
    std = self.marginal_prob_std(random_s)

    y = x * self.mu(random_s)[..., None, None, None] + z * std[..., None, None, None]
    etheta = self.neural_core.apply(params, y, c, random_s)
    prefactor = (0.5 * (self.sigma_at(random_s)/std) ** 2) ** int(self.maxL_prefactor) # Maximum likelihood.

    KL_bound = jnp.sum(((self.grad_logp_eq(y, random_s) + etheta) * std[..., None, None, None] + z) ** 2, axis=(1,2,3)) # Variance dropped OG loss, shape is [batch_size * num_steps]
    return jnp.mean(prefactor * KL_bound)
  

#------------------------------------------------------------------------
# The conditional entropy matching model (VPx), from NeuralEntropyRedux/CIFAR-10.
class DiffusionImagesEMConditional(DiffusionImagesConditional):
  def __init__(self, neural_core, beta_min, beta_max, kappa=1., maxL_prefactor=False):
    """
    Initializes the diffusion model with a provided neural network module
    and the diffusion process parameters.
    """
    super().__init__(neural_core, maxL_prefactor)
    self.beta_min = beta_min
    self.beta_max = beta_max
    self.kappa = kappa
  
  # VP SDE with kappa
  def beta_at(self, s):
    return self.beta_min + (self.beta_max - self.beta_min) * s
  
  def bplus(self, x, s):
    return - self.beta_at(s)[..., None, None, None] * x / 2

  def sigma_at(self, s):
    return self.kappa * jnp.sqrt(self.beta_at(s))

  # This is the square root of expintbeta.
  def mu(self, s):
    return jnp.exp(- 0.5 * self.beta_min * s - 0.25 * (self.beta_max - self.beta_min) * (s ** 2))

  def marginal_prob_std(self, s):
    return self.kappa * jnp.sqrt((1 - self.mu(s)**2))
  
  def grad_logp_eq(self, x, s):
    return - x / self.kappa**2
  

#------------------------------------------------------------------------
# The conditional entropy matching model (VP), from NeuralEntropyRedux/CIFAR-10.
# To be used with the U-net with self-attention.
class DiffusionImagesCondEM():
  def __init__(self, neural_core, classes, prior_prob, beta_min, beta_max):
    """
    Initializes the diffusion model with a provided neural network module
    and the diffusion process parameters.
    """
    self.neural_core = neural_core
    self.classes = classes
    self.prior_prob = prior_prob # p(c)
    self.num_classes = len(prior_prob)

    self.beta_min = beta_min
    self.beta_max = beta_max

  def init(self, key, x, c, s):
    D = jnp.prod(jnp.array(x.shape[-3:]))
    self.S0 = 0.5 * D * jnp.log(2 * jnp.pi * jnp.e)
    
    return self.neural_core.init(key, x, c, s, train=False)
  
  def apply(self, params, x, c, s):
    return self.neural_core.apply(params, x, c, s, train=False)

  # VP SDE
  def beta_at(self, t):
    return self.beta_min + (self.beta_max - self.beta_min) * t
  
  def bplus(self, x, t):
    return - self.beta_at(t)[..., None, None, None] * x / 2

  def sigma_at(self, t):
    return jnp.sqrt(self.beta_at(t))

  # This is the square root of expintbeta.
  def mu(self, t):
    return jnp.exp(- 0.5 * self.beta_min * t - 0.25 * (self.beta_max - self.beta_min) * (t ** 2))

  def marginal_prob_std(self, t):
    return jnp.sqrt((1 - self.mu(t)**2))
  
  def reverse_drift(self, params, x, c, t):
    return - self.beta_at(t)[..., None, None, None] / 2 * x \
      + self.sigma_at(t)[..., None, None, None] ** 2 * self.apply(params, x, c, t)

  # The OG loss for conditional generation.
  def loss(self, params, x, c, key, eps=1e-5, num_steps=1):
    x = jnp.tile(x[:, jnp.newaxis, :, :, :], (1, num_steps, 1, 1, 1)) # shape is [batch_size, num_steps, w, h, channels]
    x = x.reshape(-1, *x.shape[-3:]) # Flatten all but last 3 dimensions, so shape is [batch_size * num_steps w, h, channels]
    c = jnp.repeat(c, num_steps) # Same as jnp.tile(c[:, jnp.newaxis], (1, num_steps)).reshape(-1)
    
    # TODO: Should we be tiling random_t or using a different random t? Tiling makes sure the steps are evaluated at the same time for all classes.
    key, subkey = jax.random.split(key)
    random_t = random.uniform(subkey, (*x.shape[:-3],), minval=eps, maxval=1.) # shape is [batch_size * num_steps]
    key, subkey = jax.random.split(key)
    z = random.normal(subkey, x.shape)
    Sigma = self.marginal_prob_std(random_t)

    y = x * self.mu(random_t)[..., None, None, None] + z * Sigma[..., None, None, None]
    key, dropout_key = random.split(key)
    etheta = self.neural_core.apply(params, y, c, random_t, train=True, rngs={'dropout': dropout_key}) # Dropout is applied during training.

    # We implement the OG diffusion model loss.
    KL_bound = jnp.sum(((-y + etheta) * Sigma[..., None, None, None] + z) ** 2, axis=(1,2,3)) # Variance dropped OG loss, shape is [batch_size * num_steps]
    return jnp.mean(KL_bound)
  

#------------------------------------------------------------------------
# Evolve the Gaussian prior to the data distribution using the PF-ODE.
# Introduced in NeuralEntropyRedux.ipynb. It is dimensionn agnostic.
def transport_to_data(samples, model, score, endtime=1, num_steps=2):
    t_span = (0, endtime-0.001)
    t_eval=np.linspace(*t_span, num_steps)

    shape = samples.shape

    def derivative(t, x):
        s = endtime - t
        x = x.reshape(shape) # Put it back in the original shape to compute score.
        s_arr = jnp.ones(shape[0]) * s # Score builder expects a time value per sample.
        f = - model.bplus(x, s_arr) + 0.5 * model.sigma_at(s)**2 * score(x, s_arr)
        return f.reshape((-1,))

    x0 = samples.reshape(-1,) # Flatten
    sol = solve_ivp(derivative, t_span, x0, method='RK45', t_eval=t_eval)
    y_arr = np.transpose(sol.y).reshape(num_steps, *shape)

    return y_arr