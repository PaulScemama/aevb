try:
    import equinox as eqx
except ModuleNotFoundError:
    message = "Please install equinox to use equinox networks."

from typing import List, Callable
import jax.numpy as jnp
import jax.random as random

from jax.random import PRNGKey

class EqxMLPEncoder(eqx.Module):

    in_dim: int
    latent_dim: int
    hidden: List[int]
    activation: Callable


    def __call__(self, key, x):
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

        x = eqx.nn.Linear(self.hidden[i], self.out_dim, key=keys[i])(x)
        return x





