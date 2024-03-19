from functools import partial as bind
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Union

import jax
import jax.numpy as jnp
import optax
from jax import jit
from jax.random import PRNGKey, split
from jax.typing import ArrayLike
from optax import GradientTransformation, OptState

from jax.tree_util import tree_leaves, tree_structure, tree_unflatten

ArrayTree = Union[jax.Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
ArrayLikeTree = Union[
    ArrayLike, Iterable["ArrayLikeTree"], Mapping[Any, "ArrayLikeTree"]
]
"""
References:

    [1] "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
"""


def normal_like(rng_key: PRNGKey, tree: ArrayLikeTree, n_samples: int) -> ArrayTree:
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


def normal_loglikelihood_fn(apply_fn, params, z, x):
    """Use the `params` along with the `apply_fn` to predict `x` from `z`. Then
    compute the observed data `x` under a normal distribution parameterized by the
    predicted `x` as its mean."""
    pred_x = apply_fn(params, z)
    return -((x - pred_x) ** 2)


def reparameterized_sample(
    rng_key: PRNGKey, mu: ArrayLikeTree, sigma: ArrayLikeTree, n_samples: int
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
    rng_key: PRNGKey,
    rec_params: ArrayLikeTree,
    gen_params: ArrayLikeTree,
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

        pred_z_mu, pred_z_sigma = rec_apply_fn(rec_params, x)
        z_samples = reparameterized_sample(rng_key, pred_z_mu, pred_z_sigma, n_samples)
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
    return (new_rec_params, new_gen_params, new_opt_state), (loss_val, nll, kl)


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




class AEVBState(NamedTuple):
    rec_params: ArrayLikeTree
    gen_params: ArrayLikeTree
    opt_state: OptState


class AEVBInfo(NamedTuple):
    loss: float
    nll: float
    kl: float


def AEVB(
    latent_dim: int,
    generative_model: Callable | tuple[Callable, Callable],
    recognition_model: Callable | tuple[Callable, Callable],
    optimizer: GradientTransformation,
    n_samples: int,
) -> tuple[Callable, Callable, Callable]:
    """_summary_

    Args:
        latent_dim (int): _description_
        generative_model (tuple[Callable, Callable]): _description_
        recognition_model (tuple[Callable, Callable]): _description_
        optimizer (GradientTransformation): _description_
        n_samples (int): _description_

    Returns:
        tuple[Callable, Callable, Callable]: _description_
    """

            
    def _init_apply_init_fn(rng_key: PRNGKey, data_shape: tuple, rec_init: Callable, gen_init: Callable) -> AEVBState:
        rec_init_key, gen_init_key = split(rng_key)
        rec_params = rec_init(rec_init_key, jnp.ones(data_shape))
        gen_params = gen_init(gen_init_key, jnp.ones(latent_dim))
        opt_state = optimizer.init((rec_params, gen_params))
        return AEVBState(rec_params, gen_params, opt_state)

    def _model_init_fn(rec_params, gen_params) -> AEVBState:
        opt_state = optimizer.init((rec_params, gen_params))
        return AEVBState(rec_params, gen_params, opt_state)

    # TODO: handle different generative model / recognition models
    # gen_init, gen_apply = generative_model
    # rec_init, rec_apply = recognition_model
    if isinstance(recognition_model, tuple) and isinstance(generative_model, tuple):
        rec_init, _ = recognition_model
        gen_init, _ = generative_model
        init_fn = bind(_init_apply_init_fn, rec_init=rec_init, gen_init=gen_init)
        _, rec_apply = recognition_model
        _, gen_apply = generative_model

    elif isinstance(recognition_model, object) and isinstance(generative_model, object):
        # Passed in a Module instance with .init, .apply 
        try:
            rec_init, _ = recognition_model.init, recognition_model.apply
            gen_init, _ = generative_model.init, generative_model.apply
            init_fn = bind(_init_apply_init_fn,  rec_init=rec_init, gen_init=gen_init)
            rec_apply = recognition_model.apply
            gen_apply = generative_model.apply
        # Passed in a Module instance without a .init, .apply method
        except AttributeError:
            init_fn = _model_init_fn
            rec_apply = recognition_model
            gen_apply = generative_model

    @jit
    def step_fn(rng_key, state, x) -> tuple[AEVBState, AEVBInfo]:
        """Take a step of Algorithm 1 from [1] using the second verison of the
        SGVB estimator which takes advantage of an analytical KL term when the prior
        p(z) is a unit normal.

        Args:
            rng_key (PRNGKey): Random number generator key.
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
        (new_rec_params, new_gen_params, new_opt_state), (loss_val, nll, kl) = (
            tractable_kl_step(
                rng_key,
                state.rec_params,
                state.gen_params,
                state.opt_state,
                x,
                rec_apply,
                gen_apply,
                optimizer,
                n_samples,
            )
        )
        return AEVBState(new_rec_params, new_gen_params, new_opt_state), AEVBInfo(
            loss_val, nll, kl
        )

    @bind(jit, static_argnames=["n_samples"])
    def sample_data_fn(rng_key, gen_params, n_samples) -> ArrayLike:
        """_summary_

        Args:
            rng_key (_type_): _description_
            gen_params (_type_): _description_
            n_samples (_type_): _description_

        Returns:
            ArrayLike: _description_
        """
        return sample_data(
            rng_key, gen_params, gen_apply, n_samples, latent_dim
        )

    return init_fn, step_fn, sample_data_fn
