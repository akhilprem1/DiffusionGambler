# Functions to make Gaussian mixture distributions and their score functions.
# The score builder was introduced in DistilledClassifier.ipynb and refined in NeuralEntropyRedux.ipynb.

import jax
import jax.numpy as jnp
from jax import jit, random
import distrax

# A function to make Gaussian mixture distributions.
# The means are restricted to a cube of side length box_size.
# The standard deviations are in the interval [0.5, 1] * scale.
def make_distribution_flexible(key, D, n_r, box_size=4, scale=1):
    mus = random.uniform(key, shape=(n_r, D), minval=-box_size/2, maxval=box_size/2) # IMPORTANT: Changed randint to uniform
    sigmas = jnp.ones(shape=(n_r, D)) * random.uniform(key, shape=(n_r,), minval=0.5*scale, maxval=scale)[:,None]

    mixing_probs = random.uniform(key, shape=(n_r,))
    mixing_probs = mixing_probs / jnp.sum(mixing_probs)

    # Create the Gaussian components and combine them into a distribution
    components = distrax.MultivariateNormalDiag(loc=mus, scale_diag=sigmas)
    dist = distrax.MixtureSameFamily(mixture_distribution=distrax.Categorical(probs=mixing_probs),
                                    components_distribution=components)
    
    return dist

# This function creates a *single* intermediate distribution at a particular
# instant of time s and computes the weights w_i(x,s) of which there are num_components.
# We vectorize this function later to apply it on several instants of time simultaneosuly.
# int_m and int_var has shape [num_components,D], int_m has shape [num_components, D].
# Historical note: At one point we tried to return an array of Gaussian mixtures, but
# vmap combined them into one large distribution with len(s) * num_components Gaussians.
def mixing_w(mixing_probs, int_m, int_std, x):
    mixing_probs = mixing_probs / jnp.sum(mixing_probs)

    # Create the individual Gaussian components
    components = distrax.MultivariateNormalDiag(
        loc=int_m, scale_diag=int_std
    )

    # Create the Gaussian mixture distribution
    gmm = distrax.MixtureSameFamily(
        mixture_distribution=distrax.Categorical(probs=mixing_probs),
        components_distribution=components
    )

    # x has shape [D,]. We extend this to [None, D] so that the dimensions
    # of x are aligned with the components of the distribution.
    # Think of this as a function p_i(x^{i...}) which gives the probability of x
    # the i-th component.
    f = - gmm.components_distribution.log_prob(x[None, :])
    f_star = jnp.min(f)
    w = mixing_probs * jnp.exp(-(f-f_star))
    w = w / jnp.sum(w) # shape is [num_components,]

    return w


def score_builder(dist, model):
    # Due to lexical scoping in Python, the values of dist, model, r, m, h is what the score function below uses.
    r = dist.mixture_distribution.probs
    m = dist.components_distribution.loc # shape is [num_components, D]
    h = dist.components_distribution.scale_diag # [num_components,D]

    # x, u have shape [s.shape, D]
    def scoreGM(x, s):
        # We flatten s, x, and u to have leading dimension be len(s) so that s has shape [len(s)] and
        # x, u have shape [len(s), D]. Keep in mind that vmap can only map over one axes, which is why
        # we limit ourselves to a one dimensional s. The original shape of s is restored at the very end.

        s_shape = s.shape
        s = s.reshape(-1)
        x = x.reshape(-1, x.shape[-1])

        # Means with exponential factor in time.
        int_means = m[None, ...] * model.mu(s)[..., None, None] # shape is [len(s), num_components, D]
        int_vars = (h[None, ...] * model.mu(s)[..., None, None])**2 + model.marginal_prob_std(s)[..., None, None]**2
        int_stds = jnp.sqrt(int_vars)

        mixing_w_vectorized = jax.vmap(mixing_w, in_axes=(None, 0, 0, 0))
        w = mixing_w_vectorized(r, int_means, int_stds, x) # shape is [len(s), num_components]

        # It might be better to build the w in the same function where we make gmm to avoid the coalescing problem.

        # x : [len(s), D] --> x[..., None, :]: [len(s), num_components, D]
        score_comp = -(x[..., None, :] - int_means) / int_vars 

        # w: [len(s), num_components] to [len(s), num_components, D].
        # The result is contracted along the num_components axis.
        score = jnp.sum(score_comp * w[..., None], axis=1) # has same shape as x i.e. [len(s), D].

        return score.reshape(s_shape + (score.shape[-1],)) # shape is [s.shape, D]
    
    return scoreGM