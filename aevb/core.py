from typing import Any, Callable, Iterable, Mapping, Type

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.typing import ArrayLike
from optax import GradientTransformation

from aevb._src.core import AEVB as _AEVB
from aevb._src.core import (AEVBAlgorithm, AEVBAlgorithmUtil, AEVBInfo,
                            AEVBState)

ArrayTree = jax.Array | Iterable["ArrayTree"] | Mapping[Any, "ArrayTree"]
ArrayLikeTree = ArrayLike | Iterable["ArrayLikeTree"] | Mapping[Any, "ArrayLikeTree"]

from dataclasses import replace


def AEVB(
    latent_dim: int,
    generative_model: object,
    recognition_model: object,
    optimizer: GradientTransformation,
    n_samples: int,
    nn_lib: str = "flax",
) -> tuple[Callable, Callable, Callable]:
    if nn_lib not in ["flax", "equinox"]:
        raise NotImplementedError(
            "Currently only support 'construction-from-model-instance' for flax and equinox. \
                                        You can 'construct-from-apply-function' using `aevb._src.core.AEVB`."
        )

    if nn_lib == "flax":
        from aevb._src.flax_util import init_apply_flax_model

        gen_init, gen_apply = init_apply_flax_model(generative_model)
        rec_init, rec_apply = init_apply_flax_model(recognition_model)

        # Override init fn.
        #
        def init_fn(rng_key, data_dim) -> AEVBState:
            (gen_params, gen_state) = gen_init(rng_key, jnp.ones((1, latent_dim)))
            (rec_params, rec_state) = rec_init(rng_key, jnp.ones((1, data_dim)))
            return _init_fn(rec_params, rec_state, gen_params, gen_state)

        _aevb_algorithm = _AEVB(latent_dim, gen_apply, rec_apply, optimizer, n_samples)

        util = replace(
            _aevb_algorithm.util,
            rec_apply=rec_apply,
            gen_apply=gen_apply,
            rec_init=rec_init,
            gen_init=gen_init,
        )
        aevb_algorithm = replace(_aevb_algorithm, init=init_fn, util=util)

    if nn_lib == "equinox":
        from aevb._src.eqx_util import init_apply_eqx_model

        # Make sure latent dim matches
        for model in [generative_model, recognition_model]:
            _eqx_latent_dim_check(model[0], latent_dim)

        gen_init, gen_apply = init_apply_eqx_model(generative_model)
        rec_init, rec_apply = init_apply_eqx_model(recognition_model)

        _init_fn, step_fn = _AEVB(gen_apply, rec_apply, optimizer, n_samples)

        # Override init fn
        def init_fn() -> AEVBState:
            (gen_params, gen_state) = gen_init()
            (rec_params, rec_state) = rec_init()
            return _init_fn(rec_params, rec_state, gen_params, gen_state)

    util = AEVBAlgorithmUtil(latent_dim, rec_apply, gen_apply, rec_init, gen_init)
    return AEVBAlgorithm(init_fn, step_fn, util=util)


def _eqx_latent_dim_check(model, latent_dim):
    if "latent_dim" in model.__annotations__.keys():
        assert (
            model.latent_dim == latent_dim
        ), f"""
                    latent_dim value passed to AEVB() does not match attribute of {model}. 
                        These need to match."""
