from flax import linen as nn
import jax
from jax import jit, random
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from typing import Optional, Tuple
import orbax
from orbax import checkpoint as orbax_checkpoint
from flax.training import orbax_utils
import flax.jax_utils as jax_utils
import os
import sys
import flax.serialization as serialization

def save_model_params(model_params, save_dir):
    """
    Saves the model parameters to a checkpoint file using Orbax.

    Args:
        model_params: The model parameters to save (e.g., Flax model parameters).
        experiment_dir: Directory where the checkpoint should be saved.
    """
    # Define the path for the model checkpoint
    save_dir = os.path.abspath(save_dir)
    model_fname = os.path.join(save_dir, "model_checkpoint")
    
    orbax_checkpointer = orbax_checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(model_params)
    
    orbax_checkpointer.save(model_fname, model_params, save_args=save_args)


def load_model_params(save_dir):
    """
    Loads model parameters from a checkpoint file located in a specified directory.

    Args:
        checkpoint_dir: The directory where the model checkpoint is saved.

    Returns:
        The restored model parameters, or None if the checkpoint does not exist.
    """
    # Construct the path to the checkpoint
    model_path = os.path.join(os.path.abspath(save_dir), "model_checkpoint")

    if not os.path.exists(model_path):
        print(f"{model_path} does not exist. Skipping.")
        return None

    orbax_checkpointer = orbax_checkpoint.PyTreeCheckpointer()

    params = orbax_checkpointer.restore(model_path)
    return params

# # ------- Routines for parallelized training with checkpoint saves -------

# Loads checkpoint from file. We need to send in a params, optimizers training state, and key because
# from_bytes requires an example structure to know how to reconstruct the data. In the beginning
# params and opt_state from the init functions will do.
def load_checkpoint(params, opt_state, key, filename):
    # Read the checkpoint file
    with open(filename, "rb") as f:
        checkpoint_bytes = f.read()
    
    # Deserialize the checkpoint bytes back into a state dictionary
    state = serialization.from_bytes({"params": params, "opt_state": opt_state, "key": key}, checkpoint_bytes)

    # Extract the parameters and optimizer state
    params = state["params"]
    opt_state = state["opt_state"]
    key = state["key"]

    return params, opt_state, key


# A function to write files safely. This way we preserve the last saved file even if SLURM interrupts during the save. 
def save_checkpoint(params, opt_state, key, filename):
    state = {
            "params": params,  
            "opt_state": opt_state,
            "key": key
        }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    temp_filename = filename + ".tmp"  # Temporary file
    
    # Save to a temporary file first
    with open(temp_filename, "wb") as f:
        f.write(serialization.to_bytes(state))
    
    # Atomically replace old checkpoint
    os.rename(temp_filename, filename)  # Atomic move (safe replacement)


# A version of the training function that runs on multiple GPUs.
def train_diffusion_with_classes(key,
                                 model,
                                 params,
                                 learning_rate,
                                 epochs,
                                 train_dataset,
                                 batch_size,
                                 num_steps=10,
                                 checkpoint_path=None,
                                 verbose=False):
    num_devices = jax.device_count()  # Number of available GPUs
    assert batch_size % num_devices == 0, "Batch size must be divisible by number of devices"
    
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    if checkpoint_path and os.path.exists(checkpoint_path):
        params, opt_state, key = load_checkpoint(params, opt_state, key, checkpoint_path)
    
    # Replicate parameters & optimizer state across GPUs
    params = jax_utils.replicate(params)
    opt_state = jax_utils.replicate(opt_state)

    def update_step(params, opt_state, x, c, key):
        loss_val, grads = jax.value_and_grad(partial(model.loss, num_steps=num_steps))(params, x, c, key)
        grads = jax.lax.pmean(grads, axis_name='batch') # Aggregate across devices
        updates, opt_state = optimizer.update(grads, opt_state) # The same gradient is applied to all devices...
        params = optax.apply_updates(params, updates) # ...so params are the same on all of them after the update.
        return params, opt_state, loss_val

    # Parallelize update_step across devices. 'batch' means pmap distributes the leading dimension.
    # pmap compiles the function like jit and executes in parallel on the devices.
    p_update_step = jax.pmap(update_step, axis_name='batch')

    losses = []

    ekey = key
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
            ekey, subkey = random.split(ekey)
            x = jnp.asarray(batch)
            c = jnp.asarray(c, dtype=jnp.int32)

            # Reshape batch for multiple GPUs
            x = x.reshape(num_devices, -1, *x.shape[1:])  
            c = c.reshape(num_devices, -1, *c.shape[1:])
            keys = random.split(subkey, num_devices)  # Create per-device keys

            params, opt_state, loss_val = p_update_step(params, opt_state, x, c, keys)

            # Aggregate loss from all devices
            loss = jax_utils.unreplicate(loss_val)
            avg_loss += loss * x.shape[0] * x.shape[1]
            num_items += x.shape[0] * x.shape[1]

        epoch_loss = avg_loss / num_items
        losses.append(epoch_loss)
        pbar.set_description("Loss: {:5f}".format(epoch_loss))
        if verbose:
            print(f"Epoch {i+1}: Loss = {epoch_loss:.5f}")
            sys.stdout.flush()  # Ensure SLURM writes it to output.txt immediately
        
        if checkpoint_path:
            save_checkpoint(jax_utils.unreplicate(params), jax_utils.unreplicate(opt_state), ekey, checkpoint_path)
    
    # Unreplicate parameters before returning
    params = jax_utils.unreplicate(params)

    return params, losses


# A version of the training function that can save checkpoints, runs on a single device, and does not class condition.
# So it's just train_diffusion() from DiffusionModel.py with checkpoints. The checkpoint is NOT saved at the end of
# each epoch like the method above, it is saved only after all epochs are done.
def train_diffusion_with_checkpoints(key,
                                     model,
                                     params,
                                     learning_rate,
                                     epochs,
                                     train_dataset,
                                     batch_size,
                                     num_steps=10,
                                     checkpoint_path=None):
    
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    if checkpoint_path and os.path.exists(checkpoint_path):
        params, opt_state, key = load_checkpoint(params, opt_state, key, checkpoint_path)

    updater = jit(optimizer.update) # Very important: jit here boosts speed by 4x!
    loss_grad_fn = jit(jax.value_and_grad(partial(model.loss, num_steps=num_steps))) # MUCH faster when jit-ed!
    applier = jit(optax.apply_updates) # Also speeds up by 80%!

    losses = []

    ekey = key
    for i in (pbar := tqdm(range(epochs), desc='train iter', leave=True)):
        avg_loss = 0
        num_items = 0

        # We specify the seed for PyTorch's randomness to obtain identical results between full training runs.
        # A new seed is used for each epoch to make sure that the order of samples is different across epochs.
        ekey, subkey = random.split(ekey)
        epoch_seed = int(subkey[0])
        torch_gen = torch.Generator().manual_seed(epoch_seed)
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch_gen, num_workers=0)

        for batch in data_loader:
            ekey, subkey = random.split(ekey)
            x = jnp.asarray(batch)
            loss_val, grads = loss_grad_fn(params, x, subkey)
            updates, opt_state = updater(grads, opt_state)
            params = applier(params, updates)
            avg_loss += loss_val * x.shape[0]
            num_items += x.shape[0]

        epoch_loss = avg_loss / num_items
        losses.append(epoch_loss)
        pbar.set_description("Loss: {:5f}".format(epoch_loss))

    # Checkpoint is saved at the end of training, NOT at the end of each epoch.
    if checkpoint_path:
        save_checkpoint(params, opt_state, ekey, checkpoint_path)

    return params, losses