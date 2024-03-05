from typing import Union, Iterable, Mapping, Any, Callable, NamedTuple
from jax.typing import ArrayLike
import jax
from jax import jit
import jax.numpy as jnp
from jax.random import PRNGKey, split
import optax
from optax import GradientTransformation, OptState
import flax.linen as nn
from functools import partial

Array = jnp.array
from jax.tree_util import tree_leaves, tree_structure, tree_unflatten

"""
This is ported from https://github.com/blackjax-devs/blackjax/blob/main/blackjax/types.py

Following the current best practice (https://jax.readthedocs.io/en/latest/jax.typing.html)
We use:
- `ArrayLike` and `ArrayLikeTree` to annotate function input,
- `Array` and `ArrayTree` to annotate function output.
"""
ArrayTree = Union[jax.Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
ArrayLikeTree = Union[
    ArrayLike, Iterable["ArrayLikeTree"], Mapping[Any, "ArrayLikeTree"]
]
"""-------------------------------------------------------------------------------"""
# Useful PyTree Utility: modified from https://github.com/google-research/google-research/blob/master/bnn_hmc/utils/tree_utils.py
# to allow for `n_samples`` to be taken.
"""
References:

    [1] "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
"""

# TODO:
# - how to ensure recognition model will produce z that
# the generative model can ingest.


class AEVBState(NamedTuple):
    rec_params: ArrayLikeTree
    gen_params: ArrayLikeTree
    opt_state: OptState


class AEVBInfo(NamedTuple):
    loss: float
    nll: float
    kl: float


def init(
    rec_params: ArrayLikeTree,
    gen_params: ArrayLikeTree,
    optimizer: GradientTransformation,
) -> AEVBState:
    opt_state = optimizer.init((rec_params, gen_params))
    return AEVBState(rec_params, gen_params, opt_state)


def normal_like_tree(
    rng_key: PRNGKey, tree: ArrayLikeTree, n_samples: int
) -> ArrayTree:
    treedef = tree_structure(tree)
    num_vars = len(tree_leaves(tree))
    all_keys = jax.random.split(rng_key, num=num_vars)
    noise = jax.tree_map(
        lambda p, k: jax.random.normal(k, shape=(n_samples,) + p.shape),
        tree,
        tree_unflatten(treedef, all_keys),
    )
    return noise


def normal_loglikelihood_fn(apply_fn, params, z, x):
    """Use the `params` along with the `apply_fn` to predict `x` from `z`. Then
    compute the observed data `x` under a normal distribution parameterized by the
    predicted `x` as its mean."""
    pred_x = apply_fn(params, z)
    return -((x - pred_x) ** 2)


def reparameterized_sample_loc_scale(
    rng_key: PRNGKey, loc: ArrayLikeTree, scale: ArrayLikeTree, n_samples: int
) -> ArrayTree:
    """Compute a sample from a loc-scale family distribution using the reparameterization trick."""
    noise = normal_like_tree(rng_key, loc, n_samples)
    samples = jax.tree_map(
        lambda loc, scale, noise: loc + scale * noise,
        loc,
        scale,
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
    rng_key: PRNGKey,
    rec_params,
    gen_params,
    opt_state,
    x: ArrayLike,
    rec_apply_fn: Callable,
    gen_apply_fn: Callable,
    optimizer: GradientTransformation,
    n_samples: int,
) -> tuple[AEVBState, AEVBInfo]:
    """_summary_

    Args:
        rng_key (PRNGKey): _description_
        aevb_state (AEVBState): _description_
        rec_apply_fn (Callable): _description_
        gen_apply_fn (Callable): _description_
        x (ArrayLike): _description_
        optimizer (GradientTransformation): _description_
        n_samples (int): _description_

    Returns:
        tuple[AEVBState, AEVBInfo]: _description_
    """

    def loss_fn(
        rec_params: ArrayLikeTree, gen_params: ArrayLikeTree
    ) -> tuple[float, tuple[float, float]]:
        pred_z_mu, pred_z_sigma = rec_apply_fn(rec_params, x)
        z_samples = reparameterized_sample_loc_scale(
            rng_key, pred_z_mu, pred_z_sigma, n_samples
        )
        z = z_samples.mean(axis=0)

        kl = unit_normal_kl(pred_z_mu, pred_z_sigma).mean()
        nll = -normal_loglikelihood_fn(gen_apply_fn, gen_params, z, x).sum()
        loss = nll + kl
        return loss, (nll, kl)

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1))
    (loss_val, (nll, kl)), (rec_grad, gen_grad) = loss_grad_fn(rec_params, gen_params)

    (rec_updates, gen_updates), new_opt_state = optimizer.update(
        (rec_grad, gen_grad),
        opt_state,
        (rec_params, gen_params),
    )
    new_rec_params = optax.apply_updates(rec_params, rec_updates)
    new_gen_params = optax.apply_updates(gen_params, gen_updates)
    new_aevb_state = AEVBState(
        new_rec_params,
        new_gen_params,
        new_opt_state,
    )
    return new_aevb_state, AEVBInfo(loss_val, nll, kl)


def sample_data(
    rng_key: PRNGKey,
    gen_params: ArrayLikeTree,
    gen_apply_fn: Callable,
    n_samples: int,
    latent_dim: int,
):
    z = jax.random.normal(rng_key, shape=(n_samples, latent_dim))
    x = gen_apply_fn(gen_params, z)
    return x


class AEVBAlgorithm(NamedTuple):
    recognition_model: nn.Module
    generative_model: nn.Module
    init: Callable
    step: Callable
    sample_data: Callable


# INTERFACE -------------------------------
class RecognitionModel(nn.Module):

    latent_dim: int
    feature_extractor: nn.Module

    @nn.compact
    def __call__(self, x):

        x = self.feature_extractor(x)
        # Project to mu, log variance
        z_mu = nn.Dense(features=self.latent_dim)(x)
        z_logvar = nn.Dense(features=self.latent_dim)(x)
        z_sigma = jnp.exp(z_logvar * 0.5)

        return z_mu, z_sigma


def construct_aevb(
    latent_dim: int,
    recognition_feature_extractor: Callable[[ArrayLikeTree, ArrayLike], ArrayLikeTree],
    generative_model: Callable[[ArrayLikeTree, ArrayLike], ArrayLike],
    optimizer: GradientTransformation,
    n_samples: int,
) -> AEVBAlgorithm:
    recognition_model: nn.Module = RecognitionModel(
        latent_dim, recognition_feature_extractor
    )

    def init_fn(rng_key: PRNGKey, data_shape: tuple) -> AEVBState:
        rec_init_key, gen_init_key = split(rng_key)
        rec_params = recognition_model.init(rec_init_key, jnp.ones(data_shape))
        gen_params = generative_model.init(gen_init_key, jnp.ones(latent_dim))
        return init(rec_params, gen_params, optimizer)

    @jit
    def step_fn(rng_key, state, x) -> tuple[AEVBState, AEVBInfo]:
        return tractable_kl_step(
            rng_key,
            state.rec_params,
            state.gen_params,
            state.opt_state,
            x,
            recognition_model.apply,
            generative_model.apply,
            optimizer,
            n_samples,
        )

    @partial(jit, static_argnames=["n_samples"])
    def sample_data_fn(rng_key, gen_params, n_samples) -> ArrayLike:
        return sample_data(
            rng_key, gen_params, generative_model.apply, n_samples, latent_dim
        )

    return AEVBAlgorithm(
        recognition_model, generative_model, init_fn, step_fn, sample_data_fn
    )
