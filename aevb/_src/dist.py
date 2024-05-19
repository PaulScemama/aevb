from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.stats as stats
from jax.tree_util import tree_leaves, tree_structure, tree_unflatten

from aevb._src.types import ArrayLike, ArrayLikeTree, ArrayTree

__all__ = ["Normal", "Laplace", "Logistic", "T"]


def dist_like(dist_fn: callable) -> ArrayTree:
    """Generate `n_samples` PyTree objects containing samples from a unit normal distribution."""

    def _dist_like(rng_key: random.key, tree: ArrayLikeTree, n_samples: int):
        treedef = tree_structure(tree)
        num_vars = len(tree_leaves(tree))
        all_keys = random.split(rng_key, num=num_vars)
        samples = jax.tree_map(
            lambda p, k: dist_fn(k, shape=(n_samples,) + p.shape),
            tree,
            tree_unflatten(treedef, all_keys),
        )
        return samples

    return _dist_like


def nonstandardize_loc_scale_sample(standard_sampler: callable):
    def _sample(key, loc=0, scale=1, shape=()):
        return scale * standard_sampler(key, shape=shape) + loc

    return _sample


def loc_scale_rsample(like_sampler: callable):

    def _loc_scale_rsample(
        rng_key: random.key, loc: ArrayLikeTree, scale: ArrayLikeTree, n_samples: int
    ) -> ArrayTree:
        samples = like_sampler(rng_key, loc, n_samples)
        samples = jax.tree_map(
            lambda l, s, n: l + s * n,
            loc,
            scale,
            samples,
        )
        return samples

    return _loc_scale_rsample


def nonstandardize_loc_scale_logpdf(standard_logpdf: callable):

    def _logpdf(x, loc=0, scale=1):
        logpdf_elements = jax.tree.map(standard_logpdf, x, loc, scale)  # pytree
        sum_elements = jax.tree.map(jnp.sum, logpdf_elements)  # pytree
        logpdf = jax.tree.reduce(jnp.add, sum_elements)  # float
        return logpdf

    return _logpdf


loc_scale_logpdfs = {
    "normal": nonstandardize_loc_scale_logpdf(stats.norm.logpdf),
    "laplace": nonstandardize_loc_scale_logpdf(stats.laplace.logpdf),
    "logistic": nonstandardize_loc_scale_logpdf(stats.logistic.logpdf),
    "t": nonstandardize_loc_scale_logpdf(stats.t.logpdf),
}

loc_scale_samplers = {
    "normal": nonstandardize_loc_scale_sample(random.normal),
    "laplace": nonstandardize_loc_scale_sample(random.laplace),
    "logistic": nonstandardize_loc_scale_sample(random.logistic),
    "t": nonstandardize_loc_scale_sample(random.t),
}

rsample_loc_scale_samplers = {
    "normal": loc_scale_rsample(dist_like(random.normal)),
    "laplace": loc_scale_rsample(dist_like(random.laplace)),
    "logistic": loc_scale_rsample(dist_like(random.logistic)),
    "t": loc_scale_rsample(dist_like(random.t)),
}


def _partial_but_ignore_none(fn: callable, **kwargs):
    to_set = {k: v for k, v in kwargs.items() if v is not None}
    return partial(fn, **to_set)


def construct_loc_scale_functions(name: str, loc=None, scale=None):
    out = ()
    fns = (
        loc_scale_logpdfs[name],
        loc_scale_samplers[name],
        rsample_loc_scale_samplers[name],
    )
    for fn in fns:
        out += (_partial_but_ignore_none(fn, loc=loc, scale=scale),)
    return Dist(name, *out)


class Dist(NamedTuple):
    name: str
    logpdf: callable
    sample: callable
    rsample: callable


def Normal(loc: float | ArrayLikeTree = None, scale: float | ArrayLikeTree = None):
    return construct_loc_scale_functions("normal", loc, scale)


def Laplace(loc: float | ArrayLikeTree = None, scale: float | ArrayLikeTree = None):
    return construct_loc_scale_functions("laplace", loc, scale)


def Logistic(loc: float | ArrayLikeTree = None, scale: float | ArrayLikeTree = None):
    return construct_loc_scale_functions("logistic", loc, scale)


def T(loc: float | ArrayLikeTree = None, scale: float | ArrayLikeTree = None):
    return construct_loc_scale_functions("t", loc, scale)
