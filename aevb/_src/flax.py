try:
    import flax.linen as nn 
except ModuleNotFoundError:
    message = "Please install flax to use flax networks."


from typing import List, Callable
import jax.numpy as jnp

def convert_flax_model(flax_model):
    init = convert_flax_init(flax_model.init)
    apply = convert_flax_apply(flax_model.apply)
    return init, apply

def convert_flax_apply(flax_apply):
    def apply(*, params, state={}, input, train: bool):
        out, updates = flax_apply({'params': params, **state}, input, train=train, mutable=list(state.keys()))
        return out, updates
    return apply

def convert_flax_init(flax_init):
    def init(rng_key, x):
        variables = flax_init(rng_key, x, train=False)
        params = variables['params']
        state = {k:v for k,v in variables.items() if k != 'params'}
        return params, state
    return init


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
    
