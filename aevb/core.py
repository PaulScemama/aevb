from typing import Any, Callable, Iterable, Mapping, NamedTuple, Union
from optax import GradientTransformation
import jax.numpy as jnp
from aevb._src.core import AEVBState 
from aevb._src.core import AEVBInfo
from aevb._src.core import AEVB as _AEVB


def wrap_flax_apply(flax_apply):
    def apply(*, params, state={}, input, train: bool):
        out, updates = flax_apply({'params': params, **state}, input, train=train, mutable=list(state.keys()))
        return out, updates
    return apply

def wrap_flax_init(flax_init):
    def init(rng_key, x):
        variables = flax_init(rng_key, x, train=False)
        params = variables['params']
        state = {k:v for k,v in variables.items() if k != 'params'}
        return params, state
    return init

def AEVB(
    latent_dim: int,
    generative_model,
    recognition_model: Callable,
    optimizer: GradientTransformation,
    n_samples: int,
    nn_lib: str ="flax",
) -> tuple[Callable, Callable, Callable]:
    """Create an aevb algorithm consisting of an init function,
    a step function, and a sample data function.

    Args:
        latent_dim (int): _description_
        generative_apply (Callable): _description_
        recognition_apply (Callable): _description_
        optimizer (GradientTransformation): _description_
        n_samples (int): _description_
        nn_lib (str, optional): _description_. Defaults to "flax".

    Returns:
        tuple[Callable, Callable, Callable]: _description_
    """
    if nn_lib == "flax":
        gen_apply = wrap_flax_apply(generative_model.apply)
        gen_init = wrap_flax_init(generative_model.init)

        rec_apply = wrap_flax_apply(recognition_model.apply)
        rec_init = wrap_flax_init(recognition_model.init)

        def init_fn(rng_key, data_dim, latent_dim):
            (gen_params, gen_state) = gen_init(rng_key, jnp.ones((1, latent_dim)))
            (rec_params, rec_state) = rec_init(rng_key, jnp.ones((1, data_dim)))
            opt_state = optimizer.init((rec_params, gen_params))
            return AEVBState(rec_params, rec_state, gen_params, gen_state, opt_state)

        # Override init_fn
        _, step_fn, sample_data_fn = _AEVB(
            latent_dim,
            gen_apply,
            rec_apply,
            optimizer,
            n_samples
        )

        
    return init_fn, step_fn, sample_data_fn