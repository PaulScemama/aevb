import functools

import jax
import jax.random as random
import jax.scipy.stats as stats
from jax.tree_util import tree_leaves, tree_structure, tree_unflatten

from aevb._src.types import ArrayLikeTree, ArrayTree


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


standard_samplers = {
    "normal": dist_like(random.normal),
    "laplace": dist_like(random.laplace),
    "student_t": dist_like(random.t),
    "logistic": dist_like(random.logistic),
}


def loc_scale_reparam_sample(standard_sampler: callable):

    def _loc_scale_reparam_sample(
        rng_key: random.key, loc: ArrayLikeTree, scale: ArrayLikeTree, n_samples: int
    ) -> ArrayTree:
        samples = standard_sampler(rng_key, loc, n_samples)
        samples = jax.tree_map(
            lambda l, s, n: l + s * n,
            loc,
            scale,
            samples,
        )
        return samples

    return _loc_scale_reparam_sample


def tractable_inverse_cdf_sample(): ...


def composition_sample(): ...


def loc_scale(cls, name):
    cls.param_names = ("loc", "scale")
    standard_sampler = standard_samplers[name]
    reparam_fn = loc_scale_reparam_sample(standard_sampler)
    cls.reparam_sample = reparam_fn
    return cls


def tractable_inverse_cdf(cls):
    """Class decorator adding `tractable_inverse_sample` as a method."""
    cls.reparam_sample = tractable_inverse_cdf_sample
    return cls


def composition(cls):
    """Class decorator adding `composition_sample` as a method."""
    cls.reparam_sample = composition_sample
    return cls


class normal:

    def logpdf(x, loc, scale):
        return stats.norm.logpdf(x, loc, scale).sum()

    def sample(key, loc, scale, shape=()):
        return scale * jax.random.normal(key, shape=shape) + loc

    def reparam_sample(key, loc, scale, n_samples):
        return loc_scale_reparam_sample(standard_samplers["normal"])(
            key, loc, scale, n_samples
        )


class laplace:

    def logpdf(x, loc, scale):
        return stats.laplace.logpdf(x, loc, scale).sum()
    
    def sample(key, loc, scale, shape=()):
        return scale * jax.random.laplace(key, shape=shape) + loc

    def reparam_sample(key, loc, scale, n_samples):
        return loc_scale_reparam_sample(standard_samplers["laplace"])(
            key, loc, scale, n_samples
        )

laplace = loc_scale(laplace, name="laplace")


class logistic:

    def logpdf(x, loc, scale): ...


logistic = loc_scale(logistic, name="logistic")


class student_t:

    def logpdf(x, loc, scale): ...


student_t = loc_scale(student_t, name="student_t")
