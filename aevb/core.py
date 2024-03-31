from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from optax import GradientTransformation

from aevb._src.core import AEVB as _AEVB
from aevb._src.core import AEVBAlgorithm, AEVBState

ArrayTree = jax.Array | Iterable["ArrayTree"] | Mapping[Any, "ArrayTree"]
ArrayLikeTree = ArrayLike | Iterable["ArrayLikeTree"] | Mapping[Any, "ArrayLikeTree"]


def AEVB(
    latent_dim: int,
    generative_model: object | Callable,
    recognition_model: object | Callable,
    optimizer: GradientTransformation,
    n_samples: int,
    nn_lib: str = None,
) -> AEVBAlgorithm:
    """_summary_

    Args:
        latent_dim (int): _description_
        generative_model (object | Callable): _description_
        recognition_model (object | Callable): _description_
        optimizer (GradientTransformation): _description_
        n_samples (int): _description_
        nn_lib (str, optional): _description_. Defaults to None.

    Raises:
        NotImplementedError: _description_

    Returns:
        AEVBAlgorithm: _description_
    """
    if (nn_lib is not None) and (nn_lib not in ["flax", "equinox"]):
        raise NotImplementedError(
            """Currently only support 'construction-from-model-instance' for flax and equinox. 
            You can 'construct-from-apply-function' by passing in an apply function with 
            the signature:
            
            apply(
                params: ArrayLikeTree, 
                state: ArrayLikeTree, 
                input: ArrayLike, 
                train: bool) -> (output: Any, state: ArrayLikeTree)
            """
        )
    if nn_lib is None:
        # -- GENERAL APPLY FUNCTIONS --
        # Just return the out of the core implementation: aevb._src.core.AEVB
        for model in [generative_model, recognition_model]:
            assert isinstance(model, Callable), "Setting nn_lib=None means the generative and recognition models must be callables."
        aevb_algorithm = _AEVB(
            latent_dim, generative_model, recognition_model, optimizer, n_samples
        )
        return aevb_algorithm

    if nn_lib == "flax":
        from aevb._src.flax_util import init_apply_flax_model
        from flax.linen import Module
        

        # Make sure latent dim matches
        for model in [generative_model, recognition_model]:
            assert isinstance(model, Module), "Setting nn_lib='flax' means the generative and recognition models must be flax.linen.Module instances"
            latent_dim_check(model, latent_dim)

        # -- FLAX MODULES --
        # [Step 1]
        # Convert the `Module` instance into an init and an apply function for
        # both the generative and recognition models.
        #
        # The apply functions have signature:
        #   apply(params: ArrayLikeTree, state: ArrayLikeTree, input: ArrayLike, train: bool)
        #   -> (output: Any, state: ArrayLikeTree)
        #
        # The init functions are flax specific in this case and have signature:
        #   init(key: random.key, dummy_input: Any)
        #   -> (params: ArrayLikeTree, state: ArrayLikeTree)
        gen_init, gen_apply = init_apply_flax_model(generative_model)
        rec_init, rec_apply = init_apply_flax_model(recognition_model)

        _aevb_algorithm = _AEVB(latent_dim, gen_apply, rec_apply, optimizer, n_samples)

        def init_fn(rng_key, data_dim) -> AEVBState:
            (gen_params, gen_state) = gen_init(rng_key, jnp.ones((1, latent_dim)))
            (rec_params, rec_state) = rec_init(rng_key, jnp.ones((1, data_dim)))
            return _aevb_algorithm.init(rec_params, rec_state, gen_params, gen_state)

        util = replace(
            _aevb_algorithm.util,
            rec_init=rec_init,
            gen_init=gen_init,
        )
        aevb_algorithm = replace(_aevb_algorithm, init=init_fn, util=util)
        return aevb_algorithm

    if nn_lib == "equinox":
        from aevb._src.eqx_util import init_apply_eqx_model
        from equinox.nn import Module
        # -- EQUINOX MODULES --
        # Make sure latent dim matches
        for model in [generative_model, recognition_model]:
            latent_dim_check(model[0], latent_dim)
            assert isinstance(model, Module), "Setting nn_lib='equinox' means the generative and recognition models must be equinox.linen.Module instances"

        # [Step 1]
        # Convert the `Module` instance into an init and an apply function for
        # both the generative and recognition models.
        #
        # The apply functions have signature:
        #   apply(params: ArrayLikeTree, state: ArrayLikeTree, input: ArrayLike, train: bool)
        #   -> (output: Any, state: ArrayLikeTree)
        #
        # The init functions are equinox specific in this case and have signature:
        #   init()
        #   -> (params: ArrayLikeTree, state: ArrayLikeTree)
        gen_init, gen_apply = init_apply_eqx_model(generative_model)
        rec_init, rec_apply = init_apply_eqx_model(recognition_model)

        _aevb_algorithm = _AEVB(latent_dim, gen_apply, rec_apply, optimizer, n_samples)

        def init_fn() -> AEVBState:
            (gen_params, gen_state) = gen_init()
            (rec_params, rec_state) = rec_init()
            return _aevb_algorithm.init(rec_params, rec_state, gen_params, gen_state)

        util = replace(
            _aevb_algorithm.util,
            rec_init=rec_init,
            gen_init=gen_init,
        )
        aevb_algorithm = replace(_aevb_algorithm, init=init_fn, util=util)
        return aevb_algorithm


def latent_dim_check(model, latent_dim):
    if "latent_dim" in model.__annotations__.keys():
        assert (
            model.latent_dim == latent_dim
        ), f"""
                    latent_dim value passed to AEVB() does not match attribute of {model}. 
                        These need to match."""
