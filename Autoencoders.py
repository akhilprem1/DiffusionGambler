# Some classes related to VAEs, DAEs and their training, as introduced in DiffusionClassifierRedux.ipynb.

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit, random
import optax
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from DiffusionModelImages import Dense, transport_to_data
from typing import Sequence, Tuple, Callable
import distrax
import flax.jax_utils as jax_utils
import TrainingUtil as tu
import os
import sys


# MLP based encoder and decoder definitions
class MLPEncoder(nn.Module):
    latent_dim : int
    features : list  # e.g., [512, 256]

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # flatten input
        for feat in self.features:
            x = nn.relu(nn.Dense(feat)(x))
        mu = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mu, logvar


class MLPDecoder(nn.Module):
    latent_dim: int
    features : list  # e.g., [256, 512]
    output_shape : tuple  # e.g., (28, 28, 1)

    @nn.compact
    def __call__(self, z):
        x = z
        for feat in self.features:
            x = nn.relu(nn.Dense(feat)(x))
        output_dim = np.prod(self.output_shape)
        x = nn.sigmoid(nn.Dense(output_dim)(x))
        return x.reshape((x.shape[0],) + self.output_shape)


# Convolution based encoder and decoder definitions
class ConvEncoder(nn.Module):
    latent_dim: int
    conv_features: list  # e.g., [32, 64], channel depth.
    
    @nn.compact
    def __call__(self, x):
        for feat in self.conv_features:
            x = nn.Conv(feat, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(2 * self.latent_dim)(x)  # output both mu and logvar together
        mu, logvar = jnp.split(x, 2, axis=-1)
        return mu, logvar

class ConvDecoder(nn.Module):
    latent_dim: int
    conv_features: list  # reversed version, e.g., [64, 32]
    output_shape: tuple  # e.g., (32, 32, 3)

    @nn.compact
    def __call__(self, z):
        H, W, C = self.output_shape
        # Infer final conv feature map size based on depth
        # Each Conv in encoder halves the resolution; decoder must reverse that
        factor = 2 ** len(self.conv_features) # 2^num_downsampling_steps
        Hf, Wf = H // factor, W // factor
        init_channels = self.conv_features[0]
        
        # Convert a latent_dim dimensional vector to Hf*Wf*C so upsampling produces
        # the correct output shape.
        x = nn.Dense(Hf * Wf * init_channels)(z)
        x = x.reshape((z.shape[0], Hf, Wf, init_channels))

        for feat in self.conv_features[1:]:
            x = nn.ConvTranspose(feat, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
            x = nn.relu(x)

        x = nn.ConvTranspose(C, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        return nn.sigmoid(x)  # values in [0,1]
    

class VAE(nn.Module):
    latent_dim: int
    encoder_cls: type  # e.g., MLPEncoder or ConvEncoder
    decoder_cls: type  # e.g., MLPDecoder or ConvDecoder
    encoder_kwargs: dict  # extra args for encoder
    decoder_kwargs: dict  # extra args for decoder
    kl_beta: float = 1.0

    def setup(self):
        self.encoder = self.encoder_cls(self.latent_dim, **self.encoder_kwargs)
        self.decoder = self.decoder_cls(self.latent_dim, **self.decoder_kwargs)

    def __call__(self, x, key):
        mu, logvar = self.encoder(x)
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, std.shape)
        z = mu + eps * std
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    # Note that only the batch dimension is averaged over for
    # both recon_loss and the KL term.
    def _loss(self, x, key):
        recon, mu, logvar = self(x, key)
        recon_loss = jnp.mean(jnp.sum((x - recon) ** 2, axis=(1, 2, 3)))
        kl = 0.5 * jnp.mean(jnp.sum(mu**2 + jnp.exp(logvar) - logvar - 1, axis=-1))
        return recon_loss + self.kl_beta * kl
    
    # This is what the training routine sees. We include the class label y
    # because the class labels are also passed during training.
    def loss(self, params, x, y, key):
        return self.apply(params, x, key, method=self._loss)


#-------------------------------------------------------------------
def train_vae(key, model, params, learning_rate, epochs, train_dataset, batch_size, loss_grad_fn=None):
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    updater = jit(optimizer.update)
    applier = jit(optax.apply_updates)
    if loss_grad_fn is None:
        loss_grad_fn = jit(jax.value_and_grad(model.loss))

    data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    losses = []
    ekey = key

    # Train loop
    for i in (pbar := tqdm(range(epochs), desc='train iter', leave=True)):
        avg_loss = 0
        num_items = 0

        # We specify the seed for PyTorch's randomness to obtain identical results between full training runs.
        # A new seed is used for each epoch to make sure that the order of samples is different across epochs.
        ekey, subkey = random.split(ekey)
        epoch_seed = int(subkey[0])
        torch_gen = torch.Generator().manual_seed(epoch_seed)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch_gen, num_workers=0)

        for batch, c in data_loader:
            ekey, subkey = jax.random.split(ekey)
            x = jnp.array(batch)
            y = jnp.array(c)
            loss_val, grads = loss_grad_fn(params, x, y, subkey)
            updates, opt_state = updater(grads, opt_state)
            params = applier(params, updates)
            avg_loss += loss_val * x.shape[0]
            num_items += x.shape[0]
        
        epoch_loss = avg_loss / num_items
        losses.append(epoch_loss)
        pbar.set_description("Loss: {:5f}".format(epoch_loss))

    return params, losses


#-------------------------------------------------------------------
# A simple U-net that works for MNIST and FashionMNIST.
# Adapted from NeuralCoreUnetConditional in DiffusionModelImages.py
class NeuralCoreUnetLatentConditional(nn.Module):
  channels: Tuple[int] = (32, 64, 128, 256)
  mapping_size: int = 256
  grf_scale_z: float = 10.0
  grf_scale_s: float = 30.0
  # num_classes : int = 10  # Number of classes
  # img_shape : Tuple[int] = (28, 28, 1)
  z_embed : bool = True

  @nn.compact
  def __call__(self, x, z, s):
    B_z = self.grf_scale_z * self.param('B_z', nn.initializers.normal(), (self.mapping_size, z.shape[-1]))
    B_s = self.grf_scale_s * self.param('B_s', nn.initializers.normal(), (self.mapping_size, 1))
    # Stop gradients from flowing through B_z and B_s.
    B_z = jax.lax.stop_gradient(B_z)
    B_s = jax.lax.stop_gradient(B_s)
    B_z = B_z if self.z_embed else None

    # The swish activation function
    act = nn.swish

    embed_s = self.input_mapping(s[..., None], B_s)
    embed_s = nn.Dense(self.mapping_size)(embed_s) # Conver from 2 * (mapping_size) to mapping_size
    embed_s = act(embed_s)
    
    # # Create the latent embedding also with the required mapping size
    # embed_z = nn.Embed(10, self.mapping_size)(z)
    embed_z = self.input_mapping(z, B_z)
    embed_z = nn.Dense(self.mapping_size)(embed_z)
    embed_z = act(embed_z)

    # Combine timestep and class embeddings
    embed = embed_s + embed_z
       
    # Encoding path
    # 'VALID' adds no padding. Same as leaving out the padding argument completely.
    h1 = nn.Conv(self.channels[0], (3, 3), (1, 1), padding='VALID', use_bias=False)(x)   
    ## Incorporate information from t
    h1 += Dense(self.channels[0])(embed)
    ## Group normalization
    h1 = nn.GroupNorm(4)(h1)    
    h1 = act(h1)
    h2 = nn.Conv(self.channels[1], (3, 3), (2, 2), padding='VALID', use_bias=False)(h1)
    h2 += Dense(self.channels[1])(embed)
    h2 = nn.GroupNorm()(h2)        
    h2 = act(h2)
    h3 = nn.Conv(self.channels[2], (3, 3), (2, 2), padding='VALID', use_bias=False)(h2)
    h3 += Dense(self.channels[2])(embed)
    h3 = nn.GroupNorm()(h3)
    h3 = act(h3)
    h4 = nn.Conv(self.channels[3], (3, 3), (2, 2), padding='VALID', use_bias=False)(h3)
    h4 += Dense(self.channels[3])(embed)
    h4 = nn.GroupNorm()(h4)    
    h4 = act(h4)

    # Decoding path
    h = nn.Conv(self.channels[2], (3, 3), (1, 1), padding=((2, 2), (2, 2)),
                  input_dilation=(2, 2), use_bias=False)(h4)    
    ## Skip connection from the encoding path
    h += Dense(self.channels[2])(embed)
    h = nn.GroupNorm()(h)
    h = act(h)
    h = nn.Conv(self.channels[1], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
                  input_dilation=(2, 2), use_bias=False)(
                      jnp.concatenate([h, h3], axis=-1)
                  )
    h += Dense(self.channels[1])(embed)
    h = nn.GroupNorm()(h)
    h = act(h)
    h = nn.Conv(self.channels[0], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
                  input_dilation=(2, 2), use_bias=False)(
                      jnp.concatenate([h, h2], axis=-1)
                  )    
    h += Dense(self.channels[0])(embed)
    h = nn.GroupNorm()(h)
    h = act(h)
    h = nn.Conv(1, (3, 3), (1, 1), padding=((2, 2), (2, 2)))(
        jnp.concatenate([h, h1], axis=-1)
    )

    return h
  
  # Fourier feature mapping
  def input_mapping(self, z, B):
    if B is None:
      return z
    else:
      z_proj = (2.*jnp.pi*z) @ B.T
      return jnp.concatenate([jnp.sin(z_proj), jnp.cos(z_proj)], axis=-1)


#-------------------------------------------------------------------
# Adapted from the diffusion decoder above.
class DiffusionDecoderImages(nn.Module):
  neural_cls: type  # e.g., NeuralCoreUnetConditional
  neural_kwargs: dict  # passed to neural_cls
  beta_min: float
  beta_max: float
  maxL_prefactor: bool = False
  use_dropout: bool = False
  # p_drop: float = 0.1
  loss_steps: int = 10

  def setup(self):
    # self.null_label = self.neural_kwargs.get("num_classes", None)  # optional
    self.neural_core = self.neural_cls(**self.neural_kwargs)

  def __call__(self, x, z, s):
    return self.neural_core(x, z, s)

  def beta_at(self, s):
    return self.beta_min + (self.beta_max - self.beta_min) * s

  def bplus(self, x, s):
    return - self.beta_at(s)[..., None, None, None] * x / 2

  def sigma_at(self, s):
    return jnp.sqrt(self.beta_at(s))

  def mu(self, s):
    return jnp.exp(- 0.5 * self.beta_min * s - 0.25 * (self.beta_max - self.beta_min) * (s ** 2))

  def marginal_prob_std(self, s):
    return jnp.sqrt((1 - self.mu(s)**2))

  def grad_logp_eq(self, x, s):
    return -x
  
  def replicate(self, x, z, num_steps):
    x = jnp.tile(x[:, jnp.newaxis, :, :, :], (1, num_steps, 1, 1, 1))
    x = x.reshape(-1, *x.shape[-3:]) # shape is [batch_size * num_steps, H, W, ch]
    z = jnp.tile(z[:, jnp.newaxis, :], (1, num_steps, 1))
    z = z.reshape(-1, z.shape[-1]) # shape is [batch_size * num_steps, D]

    return x, z
  
  # Evolve x to a random instant in the forward process.
  # x has shape [batch_size * num_steps, D] rather than [batch_size, num_steps, D].
  # The latter works fine for MLPs but not for CNNs. The final mean in neural entropy
  # and diffusion loss is taken over batch_size * num_steps.
  def propagate(self, x, key, eps=1e-5):
    key, subkey = random.split(key)
    random_s = random.uniform(subkey, (*x.shape[:-3],), minval=eps, maxval=1.-eps) # shape is [batch_size * num_steps]
    key, subkey = random.split(key)
    noise = random.normal(subkey, x.shape)
    std = self.marginal_prob_std(random_s)
    perturbed_x = x * self.mu(random_s)[..., None, None, None] + noise * std[..., None, None, None]

    return perturbed_x, random_s, std, noise

  def neural_entropy(self, x, z, key, num_steps=10):
    x, z = self.replicate(x, z, num_steps)
    perturbed_x, random_s, _, _ = self.propagate(x, key)

    # Equivalent to self.apply(params, *args) except that Flax takes care of
    # routing the parameters.
    etheta = self(perturbed_x, z, random_s)
    return 0.5 * jnp.mean((self.sigma_at(random_s) ** 2) * jnp.sum(etheta ** 2, axis=(1,2,3)))

  # # TODO: Update this function once we figure out null embedding for z.
  # def mutual_info(self, x, z, key, num_steps=10):
  #   x, z = self.replicate(x, z, num_steps)
  #   z_null = jnp.ones(z.shape, dtype=jnp.int32) * self.null_label
  #   perturbed_x, random_s, _, _ = self.propagate(x, key)

  #   # Equivalent to self.apply(params, *args) except that Flax takes care of
  #   # routing the parameters.
  #   etheta_cond = self(perturbed_x, z, random_s)
  #   etheta_marg = self(perturbed_x, z_null, random_s)
  #   return 0.5 * jnp.mean((self.sigma_at(random_s) ** 2) * jnp.sum((etheta_cond - etheta_marg) ** 2, axis=(1,2,3)))

  def diffusion_loss(self, x, z, key):
    # # TODO: Randomly replace some labels with a null label
    # key, subkey = random.split(key)
    # drop_mask = jax.random.bernoulli(subkey, p=self.p_drop, shape=y.shape)
    # z = jnp.where(drop_mask, self.null_label, z) # if drop_mask[i]==True: z[i] == null_label 
  
    x, z = self.replicate(x, z, self.loss_steps)
    perturbed_x, random_s, std, noise = self.propagate(x, key)

    if self.use_dropout:
      key, dropout_key = random.split(key)
      etheta = self(perturbed_x, z, random_s, train=True, rngs={'dropout': dropout_key})
    else:
      etheta = self(perturbed_x, z, random_s)

    prefactor = (0.5 * (self.sigma_at(random_s)/std) ** 2) ** int(self.maxL_prefactor)
    return jnp.mean(prefactor * jnp.sum(((-perturbed_x + etheta) * std[..., None, None, None] + noise) ** 2, axis=(1,2,3)))
  
  # The loss method that the training loop sees.
  def loss(self, params, x, z, key):
      return self.apply(params, x, z, key, method=self.diffusion_loss)
  


class DiffusionDecoderImagesLadder(DiffusionDecoderImages):
    split_time : float = 0.0
    split_dim : int = 0

    # Naive splitting. Both z_sem and z_per have to have the
    # same dimensionality.
    def __call__(self, x, z, s):
        z_sem = z[:, :self.split_dim]     # (B, D)
        z_per = z[:, self.split_dim:]      # (B, D)

        mask = (s < self.split_time)[..., None]

        # Condition on z_per at early times and on z_sem later.
        z_routed = jnp.where(mask, z_per, z_sem)
        return self.neural_core(x, z_routed, s)
    
    # # MSB splitting: Use the whole latent for s < split_time, but z_sem is frozen.
    # # For s > split_time use only z_sem.
    # def __call__(self, x, z, s):
    #     z_sem = z[:, :self.split_dim]      # (B, split_dim)
    #     z_per = z[:, self.split_dim:]      # (B, D - split_dim)

    #     # Build broadcastable mask (B, 1)
    #     mask = (s < self.split_time)[..., None]

    #     # Freeze z_sem when mask is True, keep z_per only when mask is True
    #     z_sem = jnp.where(mask, jax.lax.stop_gradient(z_sem), z_sem)
    #     z_per = jnp.where(mask, z_per, 0)

    #     # Recombine
    #     z_routed = jnp.concatenate([z_sem, z_per], axis=-1)
    #     # z_routed = z_sem + z_per # This worked better.
    #     return self.neural_core(x, z_routed, s)


#-------------------------------------------------------------------
# A 'vanilla' DAE, first introduced in DiffusionClassifierRedux..
class DiffusionAutoencoder(nn.Module):
    img_shape : Tuple[int]
    latent_dim : int
    encoder_cls : type # e.g., MLPEncoder or ConvEncoder
    encoder_kwargs : dict # extra args for the encoder
    dmodel_kwargs : dict # extra args for the decoder
    gamma : float = 1.0

    def setup(self):
        self.encoder = self.encoder_cls(self.latent_dim, **self.encoder_kwargs)
        self.dmodel = DiffusionDecoderImages(**self.dmodel_kwargs)
        
    def get_score_fn(self, z):
        score_neural = lambda x, s: self.dmodel.grad_logp_eq(x, s) + self.dmodel(x, z, s)
        return score_neural
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, key):
        z = jnp.atleast_2d(z) # shape will be (1, D) if originally (D,)
        shape = self.img_shape
        p0 = distrax.MultivariateNormalDiag(
                            loc=jnp.zeros(shape),
                            scale_diag=jnp.ones(shape) * self.dmodel.marginal_prob_std(1))
        samples_0 = p0.sample(seed=key, sample_shape=(z.shape[0],)) # shape is [1, img_shape]
        score_neural = self.get_score_fn(z)

        samples_evolution = transport_to_data(samples_0, self.dmodel, score_neural, endtime=1, num_steps=2)
        return samples_evolution[-1]

    def __call__(self, x, key):
        mu, logvar = self.encoder(x)
        std_enc = jnp.exp(0.5 * logvar)
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, std_enc.shape)
        z = mu + noise * std_enc

        key, subkey = jax.random.split(key)
        x_hat = self.decode(z, subkey)
        return x_hat, mu, logvar

    def _loss(self, x, key):
        mu, logvar = self.encoder(x)
        encoder_kl = 0.5 * jnp.mean(jnp.sum(mu**2 + jnp.exp(logvar) - logvar - 1, axis=-1))
        
        std_enc = jnp.exp(0.5 * logvar)
        key, subkey = random.split(key)
        noise = jax.random.normal(subkey, std_enc.shape)
        z = mu + noise * std_enc

        key, subkey = random.split(key)
        recon_loss = self.dmodel.diffusion_loss(x, z, subkey)

        return recon_loss + self.gamma * encoder_kl

    # This is what the training routine sees. We include the class label y
    # because the class labels are also passed during training.
    def loss(self, params, x, y, key):
        return self.apply(params, x, key, method=self._loss)


#-------------------------------------------------------------------
# This is the DiffusionSeparator class from DiffusionClassifierRedux.ipynb, with the decorretor removed.
# It is different from a vanilla DiffusionAutoencoder because we use separate encoders for the semantic
# and perceptual latents. This allows us to use a different field of view.
class DiffusionAutoencoderHeirarchical(nn.Module):
    img_shape : Tuple[int]
    latent_sem_dim : int
    latent_per_dim : int
    encoder_sem_cls : type # e.g., MLPEncoder or ConvEncoder
    encoder_per_cls : type
    encoder_sem_kwargs : dict # extra args for the encoder
    encoder_per_kwargs : dict
    dmodel_cls : type
    dmodel_kwargs : dict # extra args for the decoder
    gamma : float = 1.0

    def setup(self):
        self.encoder_sem = self.encoder_sem_cls(self.latent_sem_dim, **self.encoder_sem_kwargs)
        self.encoder_per = self.encoder_per_cls(self.latent_per_dim, **self.encoder_per_kwargs)
        self.dmodel = self.dmodel_cls(**self.dmodel_kwargs)
        
    def get_score_fn(self, z):
        score_neural = lambda x, s: self.dmodel.grad_logp_eq(x, s) + self.dmodel(x, z, s)
        return score_neural
        
    def encode(self, x):
        mu_sem, logvar_sem = self.encoder_sem(x)
        mu_per, logvar_per = self.encoder_per(x)
        mu = jnp.concatenate([mu_sem, mu_per], axis=-1)
        logvar = jnp.concatenate([logvar_sem, logvar_per], axis=-1)
        return mu, logvar

    def decode(self, z, key):
        z = jnp.atleast_2d(z) # shape will be (1, D) if originally (D,)
        shape = self.img_shape
        p0 = distrax.MultivariateNormalDiag(
                            loc=jnp.zeros(shape),
                            scale_diag=jnp.ones(shape) * self.dmodel.marginal_prob_std(1))
        samples_0 = p0.sample(seed=key, sample_shape=(z.shape[0],)) # shape is [1, img_shape]
        score_neural = self.get_score_fn(z)

        samples_evolution = transport_to_data(samples_0, self.dmodel, score_neural, endtime=1, num_steps=2)
        return samples_evolution[-1]

    def __call__(self, x, key):
        mu, logvar = self.encode(x)
        std_enc = jnp.exp(0.5 * logvar)
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, std_enc.shape)
        z = mu + noise * std_enc # This is [z_sem, z_per]

        key, subkey = jax.random.split(key)
        x_hat = self.decode(z, subkey)
        return x_hat, mu, logvar

    def _loss(self, x, key):
        mu, logvar = self.encode(x)
        encoder_kl = 0.5 * jnp.mean(jnp.sum(mu**2 + jnp.exp(logvar) - logvar - 1, axis=-1))
        
        std_enc = jnp.exp(0.5 * logvar)
        key, subkey = random.split(key)
        noise = jax.random.normal(subkey, std_enc.shape)
        z = mu + noise * std_enc # This is [z_sem, z_per]

        # The primary diffusion model that reconstructs x.
        key, subkey = random.split(key)
        recon_loss = self.dmodel.diffusion_loss(x, z, subkey)
        
        return recon_loss + self.gamma * encoder_kl

    # This is what the training routine sees. We include the class label y
    # because the class labels are also passed during training.
    def loss(self, params, x, y, key):
        return self.apply(params, x, key, method=self._loss)
    

# ------------------------------------------------------------------------
# A diffusion model to help estimate the MI between image labels and latents.
# This is DiffusionDecoder from attempt 1 in DiffusionClassifierRedux.ipynb.
# See appendix of that notebook for tests we did on this model.
class DiffusionLatent(nn.Module):
  features : Sequence[int]
  mapping_size: int
  num_dimensions : int # latent dimensions, D
  num_labels : int
  beta_min : float
  beta_max : float
  z_embed : bool = True
  maxL_prefactor : bool = False
  grf_scale_z : float = 10.0
  grf_scale_s : float = 10.0
  p_drop : float = 0.1
  loss_steps : int = 10

  def setup(self):
    self.null_label = self.num_labels  # Define null label index once here

  @nn.compact
  def __call__(self, z, y, s):
    B_z = self.grf_scale_z * self.param('B_z', nn.initializers.normal(), (self.mapping_size, self.num_dimensions))
    B_s = self.grf_scale_s * self.param('B_s', nn.initializers.normal(), (self.mapping_size, 1))
    # Stop gradients from flowing through B_z and B_s.
    B_z = jax.lax.stop_gradient(B_z)
    B_s = jax.lax.stop_gradient(B_s)
    B_z = B_z if self.z_embed else None

    act = nn.sigmoid # Todo: consider swish.

    # Create the class embedding also with the required mapping size
    # 0 to num_labels-1 denote the actual labels, and num_labels is the null label.
    class_embedding = nn.Embed(self.num_labels + 1, self.mapping_size)(y)

    embed_s = self.input_mapping(s[..., None], B_s)
    embed_s = nn.Dense(self.mapping_size)(embed_s) # Convert from 2 * (mapping_size) to mapping_size
    embed_s = act(embed_s)
    embed = embed_s + class_embedding

    pos = self.input_mapping(z, B_z)
    pos = nn.Dense(pos.shape[-1])(pos) # This definitely helps improve learned scores.
    pos = act(pos)
    h = pos

    for feat in self.features[:-1]:
        tau = nn.Dense(feat)(embed)
        h = nn.Dense(feat)(h)
        h += tau
        h = nn.LayerNorm()(h)
        h = act(h)

    h = nn.Dense(self.num_dimensions)(h)
    return h

  # Fourier feature mapping
  def input_mapping(self, z, B):
    if B is None:
      return z
    else:
      z_proj = (2.*jnp.pi*z) @ B.T
      return jnp.concatenate([jnp.sin(z_proj), jnp.cos(z_proj)], axis=-1)

  def beta_at(self, s):
    return self.beta_min + (self.beta_max - self.beta_min) * s

  def bplus(self, z, s):
    return - self.beta_at(s)[..., None] * z / 2

  def sigma_at(self, s):
    return jnp.sqrt(self.beta_at(s))

  def mu(self, s):
    return jnp.exp(- 0.5 * self.beta_min * s - 0.25 * (self.beta_max - self.beta_min) * (s ** 2))

  def marginal_prob_std(self, s):
    return jnp.sqrt((1 - self.mu(s)**2))

  def grad_logp_eq(self, z, s):
    return -z
  
  # Evolve z to a random instant in the forward process.
  # z has shape [batch_size * num_steps, D] rather than [batch_size, num_steps, D].
  # The latter works fine for MLPs but not for CNNs, so we have used the former to
  # allow for the possibility of using a CNN core for the diffusion model later.
  # The final mean in neural entropy and diffusion loss is taken over
  # batch_size * num_steps.
  def propagate(self, z, key, eps=1e-5):
    key, subkey = random.split(key)
    random_s = random.uniform(subkey, z.shape[:-1]) * (1. - eps) + eps
    key, subkey = random.split(key)
    noise = random.normal(subkey, z.shape)
    std = self.marginal_prob_std(random_s)
    perturbed_z = z * self.mu(random_s)[..., None] + noise * std[..., None]

    return perturbed_z, random_s, std, noise

  def neural_entropy(self, z, y, key, num_steps=10):
    z = jnp.tile(z[:, jnp.newaxis, :], (1, num_steps, 1))
    z = z.reshape(-1, z.shape[-1])
    y = jnp.repeat(y, num_steps) # Same as jnp.tile(c[:, jnp.newaxis], (1, num_steps)).reshape(-1)
    perturbed_z, random_s, _, _ = self.propagate(z, key)

    # Equivalent to self.apply(params, *args) except that Flax takes care of
    # routing the parameters.
    etheta = self(perturbed_z, y, random_s)
    return 0.5 * jnp.mean((self.sigma_at(random_s) ** 2) * jnp.sum(etheta ** 2, axis=-1))

  def mutual_info(self, z, y, key, num_steps=10):
    z = jnp.tile(z[:, jnp.newaxis, :], (1, num_steps, 1))
    z = z.reshape(-1, z.shape[-1])
    y = jnp.repeat(y, num_steps) # Same as jnp.tile(c[:, jnp.newaxis], (1, num_steps)).reshape(-1)
    y_null = jnp.ones(y.shape, dtype=jnp.int32) * self.null_label
    perturbed_z, random_s, _, _ = self.propagate(z, key)

    # Equivalent to self.apply(params, *args) except that Flax takes care of
    # routing the parameters.
    etheta_cond = self(perturbed_z, y, random_s)
    etheta_marg = self(perturbed_z, y_null, random_s)
    return 0.5 * jnp.mean((self.sigma_at(random_s) ** 2) * jnp.sum((etheta_cond - etheta_marg) ** 2, axis=-1))

  def diffusion_loss(self, z, y, key):
    # Randomly replace some labels with a null label
    key, subkey = random.split(key)
    drop_mask = jax.random.bernoulli(subkey, p=self.p_drop, shape=y.shape)
    y = jnp.where(drop_mask, self.null_label, y) # if drop_mask[i]==True: y[i] == null_label 
  
    z = jnp.tile(z[:, jnp.newaxis, :], (1, self.loss_steps, 1))
    z = z.reshape(-1, z.shape[-1])
    y = jnp.repeat(y, self.loss_steps)
    perturbed_z, random_s, std, noise = self.propagate(z, key)

    etheta = self(perturbed_z, y, random_s)
    prefactor = (0.5 * (self.sigma_at(random_s)/std) ** 2) ** int(self.maxL_prefactor)
    return jnp.mean(prefactor * jnp.sum(((-perturbed_z + etheta) * std[..., None] + noise) ** 2, axis=-1))
  
  # The loss method that the training loop sees.
  def loss(self, params, z, y, key):
      return self.apply(params, z, y, key, method=self.diffusion_loss)
  

# Z,Y data to use with the above diffusion model.
class GaussianMixtureDataset(torch.utils.data.Dataset):
    def __init__(self, zs, ys):
        self.zs = zs
        self.ys = ys

    def __len__(self):
        return self.zs.shape[0]

    def __getitem__(self, idx):
        z = self.zs[idx]
        y = self.ys[idx]
        return z, y