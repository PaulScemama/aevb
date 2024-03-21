try:
    import flax.linen as nn 
except ModuleNotFoundError:
    message = "Please install flax to use flax networks."


from typing import List, Callable
import jax.numpy as jnp


class FlaxMLPEncoder(nn.Module):
    latent_dim: int
    hidden: List[int]
    activation: Callable

    @nn.compact
    def __call__(self, x):
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
    def __call__(self, x):
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = self.activation(x)
        x = nn.Dense(self.out_dim)
        return x
    
