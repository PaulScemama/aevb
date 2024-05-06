import jax
import jax.random as random

from aevb._src.types import ArrayLike
from aevb.aevb import AevbEngine, AevbState
from aevb.dist import normal


def generate_random_samples(
    key: random.key, aevb_engine: AevbEngine, aevb_state: AevbState, n_samples: int
):
    z_key, x_key = random.split(key)
    # Unit normal in this example: N(0, 1)
    prior_zs = jax.random.normal(z_key, shape=(n_samples, aevb_engine.latent_dim))
    x_params, _ = aevb_engine.gen_model.apply(
        aevb_state.dec_params, aevb_state.dec_state, prior_zs, train=False
    )
    xs = normal.sample(x_key, **x_params)
    return xs


# Encode/Decode samples using Learned Recognition and Generative Models
def encode(
    key: random.key,
    aevb_engine: AevbEngine,
    aevb_state: AevbState,
    x: ArrayLike,
    n_samples: int,
):
    rec_model = aevb_engine.rec_model
    z_params, _ = rec_model.apply(
        aevb_state.enc_params, aevb_state.enc_state, x, train=False
    )
    return rec_model.variational_dist.reparam_sample(
        key, **z_params, n_samples=n_samples
    )


def decode(
    key: random.key,
    aevb_engine: AevbEngine,
    aevb_state: AevbEngine,
    z: ArrayLike,
    n_samples: int = 1,
):
    x_params, _ = aevb_engine.gen_model.apply(
        aevb_state.dec_params, aevb_state.dec_state, z, train=False
    )
    xs = normal.sample(key, **x_params, shape=(n_samples,))
    return xs
