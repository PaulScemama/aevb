from aevb._src.util import package_available
package_available("flax", file=__file__)


from typing import Callable, List

import flax.linen as nn
import jax.numpy as jnp


def init_apply_flax_model(model):

    def init(rng_key, input):
        variables = model.init(rng_key, input, train=False)
        params = variables["params"]
        state = {k: v for k, v in variables.items() if k != "params"}
        del variables
        return params, state

    def apply(params, state, input, train: bool):
        out, updates = model.apply(
            {"params": params, **state}, input, train=train, mutable=list(state.keys())
        )
        return out, updates

    return init, apply


class EncMLP(nn.Module):
    latent_dim: int
    hidden: List[int]
    activation: Callable

    @nn.compact
    def __call__(self, x, train: bool = False):
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = self.activation(x)

        # Project to mu and log var
        mu = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        sigma = jnp.exp(logvar * 0.5)
        return mu, sigma


class DecMLP(nn.Module):
    out_dim: int
    hidden: List[int]
    activation: Callable

    @nn.compact
    def __call__(self, x, train: bool = False):
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = self.activation(x)
        x = nn.Dense(self.out_dim)
        return x
