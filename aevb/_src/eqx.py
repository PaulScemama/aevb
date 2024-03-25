try:
    import equinox as eqx
except ModuleNotFoundError:
    message = "Please install equinox to use equinox networks."

from typing import List, Callable, Type

import jax
import jax.numpy as jnp
import jax.random as random

from functools import partial as bind

from jax.random import PRNGKey



def batch_model(model):
    if "BatchNorm" in str(model.__annotations__.values()):
         # see BatchNorm: https://docs.kidger.site/equinox/api/nn/normalisation/
        return jax.vmap(model, in_axes=(0, None), out_axes=(0, None), axis_name="batch")
    else:
        return jax.vmap(model, in_axes=(0, None), out_axes=(0, None))


def convert_eqx_model(model_type: Type[eqx.Module], rng_key: PRNGKey, **init_kwargs):
    try:
            import equinox as eqx
    except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install equinox if you intend to use it.")

    # Get `static` information from model for combining in `apply` later...
    model, state = eqx.nn.make_with_state(model_type)(rng_key, **init_kwargs)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def init():
        return params, state
    

    def apply(*, params, state={}, input, train: bool): 
        model = eqx.combine(params, static)
        if not train:
            model = eqx.nn.inference_mode(model)
        batched_model = batch_model(model)
        out, updates = batched_model(input, state)
        return out, updates
    
    return init, apply


class EqxMLPEncoder(eqx.Module):

    in_dim: int
    latent_dim: int
    hidden: List[int]
    activation: Callable

    def __post_init__(self, rng_key: PRNGKey):
        ...


    #@bind(vmap, in_axes=[None, None, 0])
    def __call__(self, key: PRNGKey, x: jnp.array):
        keys = random.split(key, len(self.hidden) + 1)
        x = eqx.nn.Linear(self.in_dim, self.hidden[0], key=keys[0])(x)
        x = self.activation(x)

        i = 1
        while i < len(self.hidden):
            if i == 1:
                x = eqx.nn.Linear(self.hidden[0], self.hidden[i], key=keys[i])(x)
            else:
                x = eqx.nn.Linear(self.hidden[i-1], self.hidden[i], key=keys[i])(x)
            x = self.activation(x)
            i += 1
        
        # Project to mu and log var
        mu = eqx.nn.Linear(self.hidden[-1], self.latent_dim, key=keys[i])(x)
        logvar = eqx.nn.Linear(self.hidden[-1], self.latent_dim, key=keys[i + 1])(x)
        sigma = jnp.exp(logvar * 0.5)
        return mu, sigma



class EqxMLPDecoder(eqx.Module):

    out_dim: int
    latent_dim: int
    hidden: List[int]
    activation: Callable

    #@bind(vmap, in_axes=[None, None, 0])
    def __call__(self, key, x):
        keys = random.split(key, len(self.hidden))
        x = eqx.nn.Linear(self.latent_dim, self.hidden[0], key = keys[0])(x)
        x = self.activation(x)

        i = 1
        while i < len(self.hidden):
            if i == 1:
                x = eqx.nn.Linear(self.hidden[0], self.hidden[i], key=keys[i])(x)
            else:
                x = eqx.nn.Linear(self.hidden[i-1], self.hidden[i], key=keys[i])(x)
            x = self.activation(x)
            i += 1

        x = eqx.nn.Linear(self.hidden[-1], self.out_dim, key=keys[i])(x)
        return x





