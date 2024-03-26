try:
    import flax.linen as nn
except ModuleNotFoundError:
    message = "Please install flax to use flax networks."


from typing import Callable, List

import jax.numpy as jnp


def init_apply_flax_model(model: nn.Module):

    def init(rng_key, x):
        variables = model.init(rng_key, x, train=False)
        params = variables["params"]
        state = {k: v for k, v in variables.items() if k != "params"}
        del variables
        return params, state

    def apply(*, params, state={}, input, train: bool):
        out, updates = model.apply(
            {"params": params, **state}, input, train=train, mutable=list(state.keys())
        )
        return out, updates

    return init, apply


class FlaxMLPEncoder(nn.Module):
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


class FlaxMLPDecoder(nn.Module):
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