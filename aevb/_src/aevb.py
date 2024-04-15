from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Union

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from jax.tree_util import tree_leaves
from jax.typing import ArrayLike
from optax import GradientTransformation, OptState

from aevb._src import dist
from aevb._src.dist import normal

ArrayTree = Union[jax.Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
ArrayLikeTree = Union[
    ArrayLike, Iterable["ArrayLikeTree"], Mapping[Any, "ArrayLikeTree"]
]


builtin_priors = {"unit_normal": dist.set_params(normal, loc=0, scale=1)}

builtin_dists = {
    "normal": normal,
}


@dataclass
class GenPrior:
    logpdf: Callable
    sample: Callable = None
    name: str = None


@dataclass
class GenObsDist:
    logpdf: Callable
    sample: Callable = None


@dataclass
class RecDist:
    logpdf: Callable
    reparam_sample: Callable
    name: str = None


@dataclass
class AevbGenModel:
    prior: GenPrior
    obs_dist: GenObsDist
    apply: callable
    init: callable = None


@dataclass
class AevbRecModel:
    dist: RecDist
    apply: callable
    init: callable = None


class AevbState(NamedTuple):
    rec_params: ArrayLikeTree
    rec_state: ArrayLikeTree
    gen_params: ArrayLikeTree
    gen_state: ArrayLikeTree
    opt_state: OptState


class AevbInfo(NamedTuple):
    loss: float
    nll: float
    kl: float


class AevbEngine(NamedTuple):
    latent_dim: int
    data_dim: Union[tuple, int]
    gen_model: AevbGenModel
    rec_model: AevbRecModel

    init: Callable  # Union[Callable[[random.key, tuple[int]], AevbState], Callable[[]], AevbState]
    step: Callable[[random.key, AevbState, ArrayLike], tuple[AevbState, AevbInfo]]


def unit_normal_kl(loc, scale):
    """As per Appendix B of [1]"""

    def kl(mu, sigma):
        mu_squared = jnp.square(mu)
        sigma_squared = jnp.square(sigma)
        return -(1 + jnp.log(sigma_squared) - mu_squared - sigma_squared) / 2

    kl_tree = jax.tree_map(kl, loc, scale)
    kl_val = sum([param_kl.sum() for param_kl in tree_leaves(kl_tree)])
    return kl_val


def _step(
    rng_key: random.key,
    rec_params: ArrayLikeTree,
    rec_state: ArrayLikeTree,
    gen_params: ArrayLikeTree,
    gen_state: ArrayLikeTree,
    opt_state: ArrayLikeTree,
    x: ArrayLike,
    rec_apply: Callable,
    rec_sample: Callable,
    gen_apply: Callable,
    gen_logpdf: Callable,
    kl_fn: Callable,
    optimizer: GradientTransformation,
    n_samples: int,
) -> tuple[
    tuple[ArrayLikeTree, ArrayLikeTree, ArrayLikeTree], tuple[float, float, float]
]:
    def loss_fn(
        rec_params: ArrayLikeTree, gen_params: ArrayLikeTree
    ) -> tuple[float, tuple[float, float]]:

        z_params, upd_rec_state = rec_apply(rec_params, rec_state, x, train=True)
        z = rec_sample(rng_key, n_samples=n_samples, **z_params).mean(0)

        kl = kl_fn(**z_params)

        x_params, upd_gen_state = gen_apply(gen_params, gen_state, z, train=True)
        nll = -gen_logpdf(x, **x_params)

        loss = nll + kl
        return loss, ((nll, kl), (upd_rec_state, upd_gen_state))

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1))
    (loss_val, ((nll, kl), (rec_state, gen_state))), (rec_grad, gen_grad) = (
        loss_grad_fn(rec_params, gen_params)
    )

    (rec_updates, gen_updates), opt_state = optimizer.update(
        (rec_grad, gen_grad),
        opt_state,
        (rec_params, gen_params),
    )
    rec_params = optax.apply_updates(rec_params, rec_updates)
    gen_params = optax.apply_updates(gen_params, gen_updates)
    return (
        (rec_params, rec_state),
        (gen_params, gen_state),
        opt_state,
    ), (loss_val, nll, kl)


def make_step(
    gen_apply: Callable,
    gen_logpdf: Callable,
    rec_apply: Callable,
    rec_sample: Callable,
    kl_fn: Callable,
    optimizer: GradientTransformation,
    n_samples: int,
) -> Callable[[random.key, AevbState, jnp.array], tuple[AevbState, AevbInfo]]:
    def step(rng_key, aevb_state, x) -> tuple[AevbState, AevbInfo]:
        (
            (rec_params, rec_state),
            (gen_params, gen_state),
            opt_state,
        ), (loss_val, nll, kl) = _step(
            rng_key,
            aevb_state.rec_params,
            aevb_state.rec_state,
            aevb_state.gen_params,
            aevb_state.gen_state,
            aevb_state.opt_state,
            x,
            rec_apply,
            rec_sample,
            gen_apply,
            gen_logpdf,
            kl_fn,
            optimizer,
            n_samples,
        )
        return AevbState(
            rec_params, rec_state, gen_params, gen_state, opt_state
        ), AevbInfo(loss_val, nll, kl)

    return step


def _convert_gen_prior(dist: Union[str, Callable]) -> GenPrior:
    if isinstance(dist, str):
        name = dist
        dist = builtin_priors[dist]
        logpdf, sample = dist.logpdf, dist.sample
    elif isinstance(dist, tuple[Callable, Callable]):
        name = None
        logpdf, sample = dist
    elif isinstance(dist, Callable):
        name = None
        logpdf, sample = dist, None
    else:
        raise ValueError  # TODO: fill in
    return GenPrior(logpdf, sample, name)


def _convert_gen_obs_dist(
    dist: Union[str, Callable, tuple[Callable, Callable]]
) -> GenObsDist:
    if isinstance(dist, str):
        dist = builtin_dists[dist]
        logpdf, sample = dist.logpdf, dist.sample
    elif isinstance(dist, tuple[Callable, Callable]):
        logpdf, sample = dist
    elif isinstance(dist, Callable):
        logpdf, sample = dist, None
    else:
        raise ValueError  # TODO: fill in
    return GenObsDist(logpdf, sample)


def _convert_rec_dist(dist: Union[str, tuple[Callable, Callable]]) -> RecDist:
    if isinstance(dist, str):
        dist_name = dist
        dist = builtin_dists[dist]
        logpdf, reparam_sample = dist.logpdf, dist.reparam_sample
    elif isinstance(dist, tuple[Callable, Callable]):
        dist_name = None
        logpdf, reparam_sample = dist
    else:
        raise ValueError  # TODO: fill in
    return RecDist(logpdf, reparam_sample, dist_name)


def _create_kl_fn(prior: GenPrior, dist: RecDist) -> Callable:
    if prior.name == "unit_normal" and dist.name == "normal":
        return unit_normal_kl
    else:
        raise NotImplementedError  # TODO: fill in


class Aevb:

    convert_gen_prior = staticmethod(_convert_gen_prior)
    convert_gen_obs_dist = staticmethod(_convert_gen_obs_dist)
    convert_rec_dist = staticmethod(_convert_rec_dist)
    create_kl_fn = staticmethod(_create_kl_fn)
    make_step = staticmethod(make_step)

    def __new__(
        cls,
        latent_dim: int,
        data_dim: int,
        gen_prior: Union[str, Callable, tuple[Callable, Callable]],
        gen_obs_dist: Union[str, Callable, tuple[Callable, Callable]],
        gen_apply: Callable,
        gen_init: Callable,
        rec_dist: Union[str, tuple[Callable, Callable]],
        rec_apply: Callable,
        rec_init: Callable,
        optimizer: GradientTransformation,
        n_samples: int,
    ):
        gen_prior: GenPrior = cls.convert_gen_prior(gen_prior)
        gen_obs_dist: GenObsDist = cls.convert_gen_obs_dist(gen_obs_dist)
        rec_dist: RecDist = cls.convert_rec_dist(rec_dist)
        kl_fn: Callable = cls.create_kl_fn(gen_prior, rec_dist)

        gen_model: AevbGenModel = AevbGenModel(
            prior=gen_prior,
            obs_dist=gen_obs_dist,
            apply=gen_apply,
            init=gen_init,
        )
        rec_model: AevbRecModel = AevbRecModel(
            dist=rec_dist, init=rec_init, apply=rec_apply
        )

        def aevb_init(gen_init_args, rec_init_args) -> AevbState:
            gen_params, gen_state = gen_init(*gen_init_args)
            rec_params, rec_state = rec_init(*rec_init_args)
            opt_state = optimizer.init((rec_params, gen_params))
            return AevbState(rec_params, rec_state, gen_params, gen_state, opt_state)

        aevb_step = cls.make_step(
            gen_apply,
            gen_obs_dist.logpdf,
            rec_apply,
            rec_dist.reparam_sample,
            kl_fn,
            optimizer,
            n_samples,
        )

        return AevbEngine(
            latent_dim, data_dim, gen_model, rec_model, aevb_init, aevb_step
        )
