from typing import Union, Iterable, Mapping, Any, Callable, NamedTuple
from jax.typing import ArrayLike
from attrs import define
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import optax
from optax import GradientTransformation, OptState
from functools import partial as bind

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


# Useful PyTree Utility: modified from https://github.com/google-research/google-research/blob/master/bnn_hmc/utils/tree_utils.py
# to allow for `n_samples`` to be taken.
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
    aevb_state: AEVBState,
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
    rec_params, gen_params = aevb_state.rec_params, aevb_state.gen_params

    def loss_fn(
        rec_params: ArrayLikeTree, gen_params: ArrayLikeTree
    ) -> tuple[float, tuple[float, float]]:
        pred_z_mu, pred_z_sigma = rec_apply_fn(rec_params, x)
        z = reparameterized_sample_loc_scale(
            rng_key, pred_z_mu, pred_z_sigma, n_samples
        )

        kl = unit_normal_kl(pred_z_mu, pred_z_sigma)
        nll = -normal_loglikelihood_fn(gen_apply_fn, gen_params, z, x)
        loss = (nll + kl).mean()
        return loss, (nll, kl)

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1))
    (loss_val, (nll, kl)), (rec_grad, gen_grad) = loss_grad_fn(rec_params, gen_params)

    (rec_updates, gen_updates), new_opt_state = optimizer.update(
        (rec_grad, gen_grad),
        aevb_state.opt_state,
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


def _sample(gen_params: ArrayLikeTree, gen_apply_fn: Callable):
    ...


class InferenceAlgorithm(NamedTuple):
    init: Callable
    step: Callable
    sample: Callable


# INTERFACE -------------------------------
@define
class AEVB:
    _init_fn = staticmethod(init)
    _step_fn = staticmethod(tractable_kl_step)
    # TODO: sample_fn: Callable[[int], ArrayLike]
    # given a number of samples, sample from generative model.
    # using z = N(0,1)

    @classmethod
    def init(
        cls,
        recognition_apply: Callable[[ArrayLikeTree, ArrayLike], ArrayLikeTree],
        generative_apply: Callable[[ArrayLikeTree, ArrayLike], ArrayLike],
        optimizer: GradientTransformation,
        n_samples: int,
    ):
        init_fn = bind(cls._init_fn, optimizer=optimizer)
        step_fn = bind(
            cls._step_fn,
            rec_apply_fn=recognition_apply,
            gen_apply_fn=generative_apply,
            optimizer=optimizer,
            n_samples=n_samples,
        )
        sample_fn = None  # FOR NOW
        return InferenceAlgorithm(init_fn, step_fn, sample_fn)
