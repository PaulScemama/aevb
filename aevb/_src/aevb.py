from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from jax.tree_util import tree_leaves
from jax.typing import ArrayLike
from optax import GradientTransformation, OptState
from functools import partial

from aevb._src import dist
from aevb._src.types import ArrayLike, ArrayLikeTree


builtin_priors = {"unit_normal": dist.set_params(dist.normal, loc=0, scale=1)}

builtin_dists = {
    "normal": dist.normal,
    "laplace": dist.laplace,
    "student_t": dist.student_t,
}


class Prior(NamedTuple):
    logpdf: Callable
    name: str = None


class ObsDist(NamedTuple):
    logpdf: Callable


class VariationalDist(NamedTuple):
    logpdf: Callable
    reparam_sample: Callable
    name: str = None


class AevbGenModel(NamedTuple):
    prior: Prior
    obs_dist: ObsDist
    apply: callable
    init: callable 


class AevbRecModel(NamedTuple):
    variational_dist: VariationalDist
    apply: callable
    init: callable


class AevbState(NamedTuple):
    enc_params: ArrayLikeTree
    enc_state: ArrayLikeTree
    dec_params: ArrayLikeTree
    dec_state: ArrayLikeTree
    opt_state: OptState


class AevbInfo(NamedTuple):
    loss: float
    nll: float
    kl: float


class AevbEngine(NamedTuple):
    latent_dim: int
    data_dim: int | tuple
    gen_model: AevbGenModel
    rec_model: AevbRecModel

    init: Callable  
    step: Callable[[random.key, AevbState, ArrayLike], tuple[AevbState, AevbInfo]]


def _unit_normal_kl(_, loc, scale):
    """As per Appendix B of [1]"""

    def kl(mu, sigma):
        mu_squared = jnp.square(mu)
        sigma_squared = jnp.square(sigma)
        return -(1 + jnp.log(sigma_squared) - mu_squared - sigma_squared) / 2

    kl_tree = jax.tree_map(kl, loc, scale)
    kl_val = sum([param_kl.sum() for param_kl in tree_leaves(kl_tree)])
    return kl_val


def _approx_kl(z, prior_logpdf, variational_logpdf, **z_params):
    return variational_logpdf(z, **z_params)-prior_logpdf(z)



def _encode_and_sample(
    key: random.key,
    enc_params: ArrayLikeTree,
    enc_state: ArrayLikeTree,
    x: ArrayLike,
    enc_apply: Callable,
    variational_sample: Callable,
    n_samples: int,
):
    z_params, upd_enc_state = enc_apply(enc_params, enc_state, x, train=True)
    z = variational_sample(key, **z_params, n_samples=n_samples)
    return z, z_params, upd_enc_state


def _decode(
    dec_params: ArrayLikeTree,
    dec_state: ArrayLikeTree,
    dec_apply: Callable,
    z: ArrayLike,
):
    x_params, upd_gen_state = dec_apply(dec_params, dec_state, z, train=True)
    return x_params, upd_gen_state


def _step(
    rng_key: random.key,
    enc_params: ArrayLikeTree,
    enc_state: ArrayLikeTree,
    dec_params: ArrayLikeTree,
    dec_state: ArrayLikeTree,
    opt_state: ArrayLikeTree,
    x: ArrayLike,
    enc_apply: Callable,
    dec_apply: Callable,
    variational_sample: Callable,
    obs_logpdf: Callable,
    kl_fn: Callable,
    optimizer: GradientTransformation,
    n_samples: int,
) -> tuple[
    tuple[ArrayLikeTree, ArrayLikeTree, ArrayLikeTree], tuple[float, float, float]
]:
    
    def loss_fn(
        enc_params: ArrayLikeTree, dec_params: ArrayLikeTree
    ) -> tuple[float, tuple[float, float]]:

        # ----- Encode and Decode ----
        # z: [n_samples, batch, latent_dim]
        z, z_params, upd_enc_state = _encode_and_sample(rng_key, enc_params, enc_state, x, enc_apply, variational_sample, n_samples)

        # Each x_param is [n_samples, batch_size, ...]
        x_params, upd_dec_state = _decode(dec_params, dec_state, dec_apply, z)

        # ----- Compute losses -----
        # KL loss
        kl = kl_fn(z, **z_params)

        # x is [batch_size, data_dim]
        # Each x_param is [n_samples, batch_size, ...]
        # Broadcasting will take care of the leading dimension mismatch.
        nll = -obs_logpdf(x, **x_params) / n_samples

        # ----- Combine losses -----
        loss = nll + kl
        return loss, ((nll, kl), (upd_enc_state, upd_dec_state))

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1))
    (loss_val, ((nll, kl), (enc_state_, dec_state_))), (enc_grad, dec_grad) = (
        loss_grad_fn(enc_params, dec_params)
    )

    (enc_updates, dec_updates), opt_state_ = optimizer.update(
        (enc_grad, dec_grad),
        opt_state,
        (enc_params, dec_params),
    )
    enc_params_ = optax.apply_updates(enc_params, enc_updates)
    dec_params_ = optax.apply_updates(dec_params, dec_updates)
    return (
        (enc_params_, enc_state_),
        (dec_params_, dec_state_),
        opt_state_,
    ), (loss_val, nll, kl)


def make_step(
    enc_apply: Callable,
    dec_apply: Callable,
    obs_logpdf: Callable,
    variational_sample: Callable,
    kl_fn: Callable,
    optimizer: GradientTransformation,
    n_samples: int,
) -> Callable[[random.key, AevbState, jnp.array], tuple[AevbState, AevbInfo]]:

    # If not analytical KL, need to divide by number of samples afterward
    # as in the original paper.
    if kl_fn not in [_unit_normal_kl]:
        _kl_fn = lambda z, **z_params: kl_fn(z, **z_params) / n_samples
    else:
        _kl_fn = kl_fn

    def step(rng_key, aevb_state, x) -> tuple[AevbState, AevbInfo]:
        (
            (enc_params, enc_state),
            (dec_params, dec_state),
            opt_state,
        ), (loss_val, nll, kl) = _step(
            rng_key,
            aevb_state.enc_params,
            aevb_state.enc_state,
            aevb_state.dec_params,
            aevb_state.dec_state,
            aevb_state.opt_state,
            x,
            enc_apply,
            dec_apply,
            variational_sample,
            obs_logpdf,
            _kl_fn,
            optimizer,
            n_samples,
        )
        return AevbState(
            enc_params, enc_state, dec_params, dec_state, opt_state
        ), AevbInfo(loss_val, nll, kl)

    return step


def _convert_prior(dist: str | Callable) -> Prior:
    if isinstance(dist, str):
        name = dist
        dist = builtin_priors[dist]
        logpdf = dist.logpdf
    elif isinstance(dist, Callable):
        name = None
        logpdf = dist
    else:
        raise ValueError  # TODO: fill in
    return Prior(logpdf, name)


def _convert_obs_dist(
    dist: str | Callable 
) -> ObsDist:
    if isinstance(dist, str):
        dist = builtin_dists[dist]
        logpdf = dist.logpdf
    elif isinstance(dist, Callable):
        logpdf = dist
    else:
        raise ValueError  # TODO: fill in
    return ObsDist(logpdf)


def _convert_variational_dist(dist: str | tuple[Callable, Callable]) -> VariationalDist:
    if isinstance(dist, str):
        dist_name = dist
        dist = builtin_dists[dist]
        logpdf, reparam_sample = dist.logpdf, dist.reparam_sample
    elif isinstance(dist, tuple):
        dist_name = None
        logpdf, reparam_sample = dist
    else:
        raise ValueError  # TODO: fill in
    return VariationalDist(logpdf, reparam_sample, dist_name)


def _create_kl_fn(prior: Prior, dist: VariationalDist) -> Callable:
    if prior.name == "unit_normal" and dist.name == "normal":
        return _unit_normal_kl
    else:
        return partial(_approx_kl, prior_logpdf=prior.logpdf, variational_logpdf=dist.logpdf)


class Aevb:

    convert_prior = staticmethod(_convert_prior)
    convert_obs_dist = staticmethod(_convert_obs_dist)
    convert_variational_dist = staticmethod(_convert_variational_dist)
    create_kl_fn = staticmethod(_create_kl_fn)
    make_step = staticmethod(make_step)

    def __new__(
        cls,
        latent_dim: int,
        data_dim: int,
        enc_apply: Callable,
        enc_init: Callable,
        dec_apply: Callable,
        dec_init: Callable,
        prior: str | Callable,
        obs_dist: str | Callable,
        variational_dist: str | tuple[Callable, Callable],
        optimizer: GradientTransformation,
        n_samples: int,
    ):
        prior: Prior = cls.convert_prior(prior)
        obs_dist: ObsDist = cls.convert_obs_dist(obs_dist)
        variational_dist: VariationalDist = cls.convert_variational_dist(variational_dist)
        kl_fn: Callable = cls.create_kl_fn(prior, variational_dist)

        gen_model: AevbGenModel = AevbGenModel(
            prior=prior,
            obs_dist=obs_dist,
            apply=dec_apply,
            init=dec_init,
        )

        rec_model: AevbRecModel = AevbRecModel(
            variational_dist=variational_dist, init=enc_init, apply=enc_apply
        )

        def aevb_init(enc_init_args, dec_init_args) -> AevbState:
            enc_params, enc_state = enc_init(*enc_init_args)
            dec_params, dec_state = dec_init(*dec_init_args)
            opt_state = optimizer.init((enc_params, dec_params))
            return AevbState(enc_params, enc_state, dec_params, dec_state, opt_state)

        aevb_step = cls.make_step(
            enc_apply,
            dec_apply,
            obs_dist.logpdf,
            variational_dist.reparam_sample,
            kl_fn,
            optimizer,
            n_samples,
        )

        return AevbEngine(
            latent_dim, data_dim, gen_model, rec_model, aevb_init, aevb_step
        )
