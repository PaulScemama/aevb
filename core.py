from typing import Union, Iterable, Mapping, Any
from jax.typing import ArrayLike

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import jax.scipy.stats as stats


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


# Useful PyTree Utility: modified from https://github.com/google-research/google-research/blob/master/bnn_hmc/utils/tree_utils.py
# to allow for `n_samples`` to be taken.
def _gaussian_like_tree(
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


def _gaussian_loglikelihood_fn(apply_fn, params, batch):
    """Pass use the parameters with the inputs from batch along
    with the apply_fn to predict an output. Then compute the normal loglikelihood
    of the observed data from batch under the model with the predicted output"""
    x, y = batch
    predicted_loc = apply_fn(params, x)
    return stats.norm.logpdf(y, loc=predicted_loc)


def _reparameterized_sample_loc_scale(
    rng_key: PRNGKey, loc: ArrayLikeTree, scale: ArrayLikeTree, n_samples: int
) -> ArrayTree:
    """Compute a sample from a loc-scale family distribution using the reparameterization trick."""
    noise = _gaussian_like_tree(rng_key, loc, n_samples)
    samples = jax.tree_map(
        lambda loc, scale, noise: loc + scale * noise,
        loc,
        scale,
        noise,
    )
    return samples


def _unit_gaussian_kl(mu, sigma):
    """As per Appendix B of [1]"""

    def kl(mu, sigma):
        mu_squared = jnp.square(mu)
        sigma_squared = jnp.square(sigma)
        return (1 + jnp.log(sigma_squared) - mu_squared - sigma_squared) / 2

    kl_tree = jax.tree_map(kl, mu, sigma)
    kl_val = sum([param_kl.sum() for param_kl in tree_leaves(kl_tree)])
    return kl_val


def _analytical_kl_svgb_estimator(
    rec_params, gen_params, rec_apply_fn, gen_apply_fn, batch
):
    x, y = batch
    params = (rec_params, gen_params)

    def loss_fn(params) -> float:
        rec_params, gen_params = params

        # Predicted parameters for latent variables from recognition model
        rec_mu, rec_sigma = rec_apply_fn(rec_params, x)
        # Sample a z using its predicted loc and scale
        z = _reparameterized_sample_loc_scale(...)
        # Predicted parameter mu for predicting the data from the latent variable z
        gen_mu = gen_apply_fn(gen_params, z)

        # Compute the KL penalty between the prior over z (unit gaussian) and current
        # recognition model posterior
        # NOTE: might have to negative this
        kl_penalty = _unit_gaussian_kl(rec_mu, rec_sigma)
        # Compute reconstruction loss term
        nll = _gaussian_loglikelihood_fn(gen_apply_fn, gen_params, batch)

    return jax.grad(loss_fn)(params)


## NOTE: recognition model must prediction a mu and sigma from a data point x.
## NOTE: generative model must predict x from a sample z


# def reparameterized_sample_loc_scale(
#     rng_key: PRNGKey, params: tuple[ArrayLikeTree, ArrayLikeTree], n_samples: int
# ) -> ArrayTree:
#     loc, scale = params.loc, params.scale
#     return _reparameterized_sample_loc_scale(rng_key, loc, scale, n_samples)
