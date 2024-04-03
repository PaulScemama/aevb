from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Union

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from jax import jit
from jax.tree_util import tree_leaves, tree_structure, tree_unflatten
from jax.typing import ArrayLike
from optax import GradientTransformation, OptState

from aevb._src.util import check_package_decorator as check_package

ArrayTree = Union[jax.Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
ArrayLikeTree = Union[
    ArrayLike, Iterable["ArrayLikeTree"], Mapping[Any, "ArrayLikeTree"]
]
"""
References:

    [1] "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
"""


def normal_like(rng_key: random.key, tree: ArrayLikeTree, n_samples: int) -> ArrayTree:
    """Generate `n_samples` PyTree objects containing samples from a unit normal distribution."""
    treedef = tree_structure(tree)
    num_vars = len(tree_leaves(tree))
    all_keys = jax.random.split(rng_key, num=num_vars)
    noise = jax.tree_map(
        lambda p, k: jax.random.normal(k, shape=(n_samples,) + p.shape),
        tree,
        tree_unflatten(treedef, all_keys),
    )
    return noise


def reparameterized_sample(
    rng_key: random.key, mu: ArrayLikeTree, sigma: ArrayLikeTree, n_samples: int
) -> ArrayTree:
    """Compute a sample from a normal distribution using the reparameterization trick."""
    noise = normal_like(rng_key, mu, n_samples)
    samples = jax.tree_map(
        lambda mu, sigma, noise: mu + sigma * noise,
        mu,
        sigma,
        noise,
    )
    return samples


def unit_normal_kl(mu, sigma):
    """As per Appendix B of [1]"""

    def kl(mu, sigma):
        mu_squared = jnp.square(mu)
        sigma_squared = jnp.square(sigma)
        return -(1 + jnp.log(sigma_squared) - mu_squared - sigma_squared) / 2

    kl_tree = jax.tree_map(kl, mu, sigma)
    kl_val = sum([param_kl.sum() for param_kl in tree_leaves(kl_tree)])
    return kl_val


def tractable_kl_step(
    rng_key: random.key,
    rec_params: ArrayLikeTree,
    rec_state: ArrayLikeTree,
    gen_params: ArrayLikeTree,
    gen_state: ArrayLikeTree,
    opt_state: ArrayLikeTree,
    x: ArrayLike,
    rec_apply_fn: Callable,
    gen_apply_fn: Callable,
    optimizer: GradientTransformation,
    n_samples: int,
) -> tuple[
    tuple[ArrayLikeTree, ArrayLikeTree, ArrayLikeTree], tuple[float, float, float]
]:
    def loss_fn(
        rec_params: ArrayLikeTree, gen_params: ArrayLikeTree
    ) -> tuple[float, tuple[float, float]]:

        (z_mu, z_sigma), new_rec_state = rec_apply_fn(
            params=rec_params, state=rec_state, input=x, train=True
        )
        z_samples = reparameterized_sample(rng_key, z_mu, z_sigma, n_samples)
        z = z_samples.mean(axis=0)
        kl = unit_normal_kl(z_mu, z_sigma).mean()

        x_pred, new_gen_state = gen_apply_fn(
            params=gen_params, state=gen_state, input=z, train=True
        )
        nll = ((x - x_pred) ** 2).sum()

        loss = nll + kl
        return loss, ((nll, kl), (new_rec_state, new_gen_state))

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1))
    (loss_val, ((nll, kl), (new_rec_state, new_gen_state))), (rec_grad, gen_grad) = (
        loss_grad_fn(rec_params, gen_params)
    )

    (rec_updates, gen_updates), new_opt_state = optimizer.update(
        (rec_grad, gen_grad),
        opt_state,
        (rec_params, gen_params),
    )
    new_rec_params = optax.apply_updates(rec_params, rec_updates)
    new_gen_params = optax.apply_updates(gen_params, gen_updates)
    return (
        (new_rec_params, new_rec_state),
        (new_gen_params, new_gen_state),
        new_opt_state,
    ), (loss_val, nll, kl)


# --- Types used in interface --- #
class AEVBState(NamedTuple):
    rec_params: ArrayLikeTree
    rec_state: ArrayLikeTree
    gen_params: ArrayLikeTree
    gen_state: ArrayLikeTree
    opt_state: OptState


class AEVBInfo(NamedTuple):
    loss: float
    nll: float
    kl: float


@dataclass
class AEVBAlgorithmUtil:
    latent_dim: int
    # `apply` always takes in (params, state, input, train) and always returns the output and updated
    # state. NOTE that state can be an empty dictionary if it is not needed.
    gen_apply: Callable[
        [ArrayLikeTree, ArrayLikeTree, ArrayLike, bool], tuple[ArrayTree, ArrayTree]
    ]
    rec_apply: Callable[
        [ArrayLikeTree, ArrayLikeTree, ArrayLike, bool], tuple[ArrayTree, ArrayTree]
    ]

    # `rec_init` and `gen_init` either takes in (rng_key, input) in the case of flax or nothing in the case of equinox.
    # It always returns a tuple of (params, state)
    gen_init: (
        Callable[[random.key, ArrayLike], tuple[ArrayTree, ArrayTree]]
        | Callable[[], tuple[ArrayTree, ArrayTree]]
    ) = None
    rec_init: (
        Callable[[random.key, ArrayLike], tuple[ArrayTree, ArrayTree]]
        | Callable[[], tuple[ArrayTree, ArrayTree]]
    ) = None

    def sample_data(self, key: random.key, aevb_state: AEVBState, n_samples: int):
        z = jax.random.normal(key, shape=(n_samples, self.latent_dim))
        # Don't need to return state as it is not updated (train=False).
        x, _ = self.gen_apply(
            aevb_state.gen_params, aevb_state.gen_state, z, train=False
        )
        return x

    def encode(
        self, key: random.key, aevb_state: AEVBState, x: ArrayLike, n_samples: int
    ):
        # Don't need to return state as it is not updated (train=False).
        (z_mu, z_sigma), _ = self.rec_apply(
            aevb_state.rec_params, aevb_state.rec_state, x, train=False
        )
        z_samples = reparameterized_sample(key, z_mu, z_sigma, n_samples)
        return z_samples

    def decode(self, aevb_state: AEVBState, z: ArrayLike):
        # Don't need to return state as it is not updated (train=False).
        x, _ = self.gen_apply(
            aevb_state.gen_params, aevb_state.gen_state, z, train=False
        )
        return x


@dataclass
class AEVBAlgorithm:
    # Either takes in (rng_key, input) in the case of flax or nothing in the case of equinox.
    # It always returns a tuple of (params, state)
    init: (
        Callable[[random.key, ArrayLike], tuple[ArrayTree, ArrayTree]]
        | Callable[[], tuple[ArrayTree, ArrayTree]]
    )

    # Takes in an rng_key, a AEVBState, and data to return an updated AEVBState and AEVBInfo.
    step: Callable[[random.key, AEVBState, ArrayLike], tuple[AEVBState, AEVBInfo]]

    util: AEVBAlgorithmUtil


# --- Functions used to interface with core functionality --- #
def latent_dim_check(model, latent_dim):
    if "latent_dim" in model.__annotations__.keys():
        assert (
            model.latent_dim == latent_dim
        ), f"""
                    latent_dim value passed to AEVB() does not match attribute of {model}. 
                        These need to match."""


def make_step_fn(
    gen_apply: Callable,
    rec_apply: Callable,
    optimizer: GradientTransformation,
    n_samples: int,
):
    @jit
    def step_fn(rng_key, aevb_state, x) -> tuple[AEVBState, AEVBInfo]:
        """Take a step of Algorithm 1 from [1] using the second version of the
        SGVB estimator which takes advantage of an analytical KL term when the prior
        p(z) is a unit normal.

        Args:
            rng_key (random.key): Random number generator key.
            rec_params (ArrayLikeTree): The current recognition model parameters.
            gen_params (ArrayLikeTree): The current generative model parameters.
            opt_state (ArrayLikeTree): The current optimizer state.
            x (ArrayLike): A mini-batch of data.
            rec_apply_fn (Callable): The recognition model apply function.
            gen_apply_fn (Callable): The generative model apply function.
            optimizer (GradientTransformation): An optax optimizer.
            n_samples (int): Number of samples to take from q(z|x).

        Returns:
            tuple[AEVBState, AEVBInfo]: _description_
        """
        (
            (new_rec_params, new_rec_state),
            (new_gen_params, new_gen_state),
            new_opt_state,
        ), (loss_val, nll, kl) = tractable_kl_step(
            rng_key,
            aevb_state.rec_params,
            aevb_state.rec_state,
            aevb_state.gen_params,
            aevb_state.gen_state,
            aevb_state.opt_state,
            x,
            rec_apply,
            gen_apply,
            optimizer,
            n_samples,
        )
        return AEVBState(
            new_rec_params, new_rec_state, new_gen_params, new_gen_state, new_opt_state
        ), AEVBInfo(loss_val, nll, kl)

    return step_fn


def setup_from_callable(
    latent_dim: int,
    gen_apply: Callable,
    rec_apply: Callable,
    optimizer: GradientTransformation,
    n_samples: int,
):
    for model in [gen_apply, rec_apply]:
        assert isinstance(
            model, Callable
        ), "Setting nn_lib=None means the generative and recognition models must be callables."

    def init_fn(rec_params, rec_state, gen_params, gen_state) -> AEVBState:
        opt_state = optimizer.init((rec_params, gen_params))
        return AEVBState(rec_params, rec_state, gen_params, gen_state, opt_state)

    step_fn = make_step_fn(gen_apply, rec_apply, optimizer, n_samples)
    util = AEVBAlgorithmUtil(latent_dim, gen_apply, rec_apply, optimizer, n_samples)
    return AEVBAlgorithm(init_fn, step_fn, util)


@check_package("flax")
def setup_from_flax_module(
    latent_dim: int,
    generative_model: object,
    recognition_model: object,
    optimizer: GradientTransformation,
    n_samples: int,
):
    from flax.linen import Module

    from aevb._src.flax_util import init_apply_flax_model

    # Make sure latent dim matches
    for model in [generative_model, recognition_model]:
        assert isinstance(
            model, Module
        ), "Setting nn_lib='flax' means the generative and recognition models must be flax.linen.Module instances"
        latent_dim_check(model, latent_dim)

    gen_init, gen_apply = init_apply_flax_model(generative_model)
    rec_init, rec_apply = init_apply_flax_model(recognition_model)

    def init_fn(rng_key, data_dim) -> AEVBState:
        (gen_params, gen_state) = gen_init(rng_key, jnp.ones((1, latent_dim)))
        (rec_params, rec_state) = rec_init(rng_key, jnp.ones((1, data_dim)))
        opt_state = optimizer.init((rec_params, gen_params))
        return AEVBState(rec_params, rec_state, gen_params, gen_state, opt_state)

    step_fn = make_step_fn(gen_apply, rec_apply, optimizer, n_samples)
    util = AEVBAlgorithmUtil(latent_dim, gen_apply, rec_apply, optimizer, n_samples)
    return AEVBAlgorithm(init_fn, step_fn, util)


@check_package("equinox")
def setup_from_equinox_module(
    latent_dim: int,
    generative_model: object,
    recognition_model: object,
    optimizer: GradientTransformation,
    n_samples: int,
):
    from equinox.nn._stateful import State

    from aevb._src.equinox_util import init_apply_eqx_model

    for model in [generative_model, recognition_model]:
        latent_dim_check(model[0], latent_dim)
        assert isinstance(
            model[1], State
        ), "Setting nn_lib='equinox' means the generative and recognition models must be the result of calling eqx.nn.make_with_state on an eqx.Module."

    gen_init, gen_apply = init_apply_eqx_model(generative_model)
    rec_init, rec_apply = init_apply_eqx_model(recognition_model)

    def init_fn() -> AEVBState:
        (gen_params, gen_state) = gen_init()
        (rec_params, rec_state) = rec_init()
        opt_state = optimizer.init((rec_params, gen_params))
        return AEVBState(rec_params, rec_state, gen_params, gen_state, opt_state)

    step_fn = make_step_fn(gen_apply, rec_apply, optimizer, n_samples)
    util = AEVBAlgorithmUtil(latent_dim, gen_apply, rec_apply, optimizer, n_samples)
    return AEVBAlgorithm(init_fn, step_fn, util)


def AEVB(
    latent_dim: int,
    generative_model: object | Callable,
    recognition_model: object | Callable,
    optimizer: GradientTransformation,
    n_samples: int,
    nn_lib: str = None,
) -> AEVBAlgorithm:
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
        gen_apply, rec_apply = generative_model, recognition_model
        return setup_from_callable(
            latent_dim, gen_apply, rec_apply, optimizer, n_samples
        )

    if nn_lib == "flax":
        return setup_from_flax_module(
            latent_dim, generative_model, recognition_model, optimizer, n_samples
        )

    if nn_lib == "equinox":
        return setup_from_equinox_module(
            latent_dim, generative_model, recognition_model, optimizer, n_samples
        )

    else:
        raise ValueError("No nn_lib conditional case was hit...")
