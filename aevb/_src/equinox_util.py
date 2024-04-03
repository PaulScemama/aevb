from aevb._src.util import check_package

check_package(__file__, "equinox")

import inspect
from typing import Any, Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
from jax.random import PRNGKey

State = eqx.nn._stateful.State


def batch_model(model: eqx.Module) -> Callable:
    if "BatchNorm" in inspect.getsource(model.__init__):
        # see BatchNorm: https://docs.kidger.site/equinox/api/nn/normalisation/
        return jax.vmap(model, in_axes=(0, None), out_axes=(0, None), axis_name="batch")
    else:
        return jax.vmap(model, in_axes=(0, None), out_axes=(0, None))


def init_apply_eqx_model(model: tuple[Any, State]) -> tuple[Callable, Callable]:
    model, state = model
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def init():
        return params, state

    def apply(params, state, input, train: bool):
        model = eqx.combine(params, static)
        if not train:
            model = eqx.nn.inference_mode(model)
        batched_model = batch_model(model)
        out, updates = batched_model(input, state)
        return out, updates

    return init, apply


@eqx.nn.make_with_state
class EncMLP(eqx.Module):

    in_dim: int
    latent_dim: int
    hidden: List[int]
    norm: tuple[Callable, List[int]]
    activation: tuple[Callable, List[int]]

    def __call__(self, key: PRNGKey, x: jnp.array):
        keys = random.split(key, len(self.hidden) + 1)
        x = eqx.nn.Linear(self.in_dim, self.hidden[0], key=keys[0])(x)
        x = self.activation(x)

        i = 1
        while i < len(self.hidden):
            if i == 1:
                x = eqx.nn.Linear(self.hidden[0], self.hidden[i], key=keys[i])(x)
            else:
                x = eqx.nn.Linear(self.hidden[i - 1], self.hidden[i], key=keys[i])(x)
            x = self.activation(x)
            i += 1

        # Project to mu and log var
        mu = eqx.nn.Linear(self.hidden[-1], self.latent_dim, key=keys[i])(x)
        logvar = eqx.nn.Linear(self.hidden[-1], self.latent_dim, key=keys[i + 1])(x)
        sigma = jnp.exp(logvar * 0.5)
        return mu, sigma


@eqx.nn.make_with_state
class DecMLP(eqx.Module):

    out_dim: int
    latent_dim: int
    hidden: List[int]
    norm: tuple[Callable, List[int]]
    activation: tuple[Callable, List[int]]

    def __call__(self, key, x):
        keys = random.split(key, len(self.hidden))
        x = eqx.nn.Linear(self.latent_dim, self.hidden[0], key=keys[0])(x)
        x = self.activation(x)

        i = 1
        while i < len(self.hidden):
            if i == 1:
                x = eqx.nn.Linear(self.hidden[0], self.hidden[i], key=keys[i])(x)
            else:
                x = eqx.nn.Linear(self.hidden[i - 1], self.hidden[i], key=keys[i])(x)
            x = self.activation(x)
            i += 1

        x = eqx.nn.Linear(self.hidden[-1], self.out_dim, key=keys[i])(x)
        return x
