import functools

import jax
import jax.random as random
import jax.scipy.stats as stats
from jax.tree_util import tree_leaves, tree_structure, tree_unflatten

from aevb._src.types import ArrayLikeTree, ArrayTree


def normal_like(rng_key: random.key, tree: ArrayLikeTree, n_samples: int) -> ArrayTree:
    """Generate `n_samples` PyTree objects containing samples from a unit normal distribution."""
    treedef = tree_structure(tree)
    num_vars = len(tree_leaves(tree))
    all_keys = random.split(rng_key, num=num_vars)
    noise = jax.tree_map(
        lambda p, k: random.normal(k, shape=(n_samples,) + p.shape),
        tree,
        tree_unflatten(treedef, all_keys),
    )
    return noise


def loc_scale_reparam_sample(
    rng_key: random.key, loc: ArrayLikeTree, scale: ArrayLikeTree, n_samples: int
) -> ArrayTree:
    noise = normal_like(rng_key, loc, n_samples)
    samples = jax.tree_map(
        lambda l, s, n: l + s * n,
        loc,
        scale,
        noise,
    )
    return samples


def tractable_inverse_cdf_sample(): ...


def composition_sample(): ...


def loc_scale(cls):
    """Class decorator adding `loc_scale_reparam_sample` as a method."""
    cls.param_names = ("loc", "scale")
    cls.reparam_sample = loc_scale_reparam_sample
    return cls


def method_list(cls):
    return [
        func
        for func in dir(cls)
        if callable(getattr(cls, func)) and not func.startswith("__")
    ]


def set_params(cls, **kwargs):
    for method in method_list(cls):
        new_method = functools.partial(getattr(cls, method), **kwargs)
        setattr(cls, method, new_method)
    return cls


def tractable_inverse_cdf(cls):
    """Class decorator adding `tractable_inverse_sample` as a method."""
    cls.reparam_sample = tractable_inverse_cdf_sample
    return cls


def composition(cls):
    """Class decorator adding `composition_sample` as a method."""
    cls.reparam_sample = composition_sample
    return cls


@loc_scale
class normal:

    def logpdf(x, loc, scale):
        return stats.norm.logpdf(x, loc, scale).sum()

    def sample(key, loc, scale, shape=()):
        return scale * jax.random.normal(key, shape=shape) + loc


@loc_scale
class laplace:

    def logpdf(x, loc, scale): ...


@loc_scale
class elliptical:

    def logpdf(x, loc, scale): ...


@loc_scale
class student_t:

    def logpdf(x, loc, scale): ...


@tractable_inverse_cdf
class exponential:
    # TODO: can we tweak this to be a part of the
    # location scale family?
    param_names = "lamb"

    def logpdf(x): ...


@tractable_inverse_cdf
class pareto:

    def logpdf(x): ...


@composition
class gamma:

    def logpdf(x): ...


@composition
class dirichlet:

    def logpdf(x): ...
