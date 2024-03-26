from typing import Callable, Type

import jax.numpy as jnp
from optax import GradientTransformation

from aevb._src.core import AEVB as _AEVB
from aevb._src.core import AEVBInfo, AEVBState


def AEVB(
    latent_dim: int,
    generative_model: object | tuple[Type, dict],
    recognition_model: object | tuple[Type, dict],
    optimizer: GradientTransformation,
    n_samples: int,
    nn_lib: str = "flax",
) -> tuple[Callable, Callable, Callable]:
    """Create an `init_fn`, `step_fn`, and `sample_data_fn` for the
    AEVB (auto-encoding variational bayes) inference algorithm. This function
    should be called using nn.Module instances from flax or equinox.

    Args:
        latent_dim (int): The dimension of the latent variable z.
        generative_apply (object): The (flax or equinox) generative model which maps a latent
        variable z to a data point x.
        recognition_apply (object): The (flax or equinox) recognition model which maps a data
        point x to its latent variable z.
        optimizer (GradientTransformation): The optax optimizer for running gradient descent.
        n_samples (int): The number of samples to take from q(z|x) for each step in the
        inference algorithm.
        nn_lib (str, optional): The neural network library used to implement the generative
        and recognition models. Defaults to "flax".

    Returns:
        tuple[Callable, Callable, Callable]: Three functions.
            1. An `init_fn`: this will output an AEVBState instance, but its arguments
            depend on whether flax or equinox is used. If flax is used, it takes an
            rng_key and the data dimension to generate the AEVBState instance. If equinox
            is used...TODO
            2. A `step_fn`: this takes in an rng_key and an AEVBState instance as
            well as a batch of data and return a new AEVBState instance and an
            AEVBInfo instance after taking a step of inference (optimization).
            3. A `sample_data_fn`: this samples datapoints x by sampling from a
            N(0,1) distribution over the latent dimension and then using the
            generative model to map these latent variable samples to data samples.
    """

    if nn_lib == "flax":
        from aevb._src.flax_util import init_apply_flax_model

        gen_init, gen_apply = init_apply_flax_model(generative_model)
        rec_init, rec_apply = init_apply_flax_model(recognition_model)

        def init_fn(rng_key, data_dim) -> AEVBState:
            (gen_params, gen_state) = gen_init(rng_key, jnp.ones((1, latent_dim)))
            (rec_params, rec_state) = rec_init(rng_key, jnp.ones((1, data_dim)))
            opt_state = optimizer.init((rec_params, gen_params))
            return AEVBState(rec_params, rec_state, gen_params, gen_state, opt_state)

    elif nn_lib == "equinox":
        # signature: (model_type: Type, key: PRNGKey, **model_kwargs)
        from aevb._src.eqx_util import init_apply_eqx_model

        def init_apply_eqx_model_args(model: tuple[Type, dict]):
            model_type, model_kwargs = model
            assert (
                "key" in model_kwargs.keys()
            ), "Need an `key`: PRNGKey key:value pair in model kwargs."
            key = model_kwargs["key"]
            model_kwargs = {k: v for k, v in model_kwargs.items() if k != "key"}
            if "latent_dim" in model_kwargs:
                assert (
                    latent_dim == model_kwargs["latent_dim"]
                ), f"""latent_dim value passed to AEVB() does not match the model kwargs passed to {model_type}. 
                    These need to match."""
            init, apply = init_apply_eqx_model(model_type, key, **model_kwargs)
            return init, apply

        gen_init, gen_apply = init_apply_eqx_model_args(generative_model)
        rec_init, rec_apply = init_apply_eqx_model_args(recognition_model)

        def init_fn() -> AEVBState:
            (gen_params, gen_state) = gen_init()
            (rec_params, rec_state) = rec_init()
            opt_state = optimizer.init((rec_params, gen_params))
            return AEVBState(rec_params, rec_state, gen_params, gen_state, opt_state)

    else:
        raise NotImplementedError(
            "Currently only support 'construction-from-model-instance' for flax and equinox. \
                                   You can 'construct-from-apply-function' using `aevb._src.core.AEVB`."
        )

    # Override init_fn
    _, step_fn, sample_data_fn = _AEVB(
        latent_dim, gen_apply, rec_apply, optimizer, n_samples
    )
    return init_fn, step_fn, sample_data_fn
