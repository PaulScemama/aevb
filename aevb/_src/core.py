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

        (z_mu, z_sigma), new_rec_state = rec_apply_fn(params=rec_params, state=rec_state, input=x, train=True)
        z_samples = reparameterized_sample(rng_key, z_mu, z_sigma, n_samples)
        z = z_samples.mean(axis=0)
        kl = unit_normal_kl(z_mu, z_sigma).mean()

        x_pred, new_gen_state = gen_apply_fn(params=gen_params, state=gen_state, input=z, train=True)
        nll = ((x - x_pred) ** 2).sum()

        loss = nll + kl
        return loss, ((nll, kl), (new_rec_state, new_gen_state))

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1))
    (loss_val, ((nll, kl), (new_rec_state, new_gen_state))), (rec_grad, gen_grad) = loss_grad_fn(rec_params, gen_params)

    (rec_updates, gen_updates), new_opt_state = optimizer.update(
        (rec_grad, gen_grad),
        opt_state,
        (rec_params, gen_params),
    )
    new_rec_params = optax.apply_updates(rec_params, rec_updates)
    new_gen_params = optax.apply_updates(gen_params, gen_updates)
    return ((new_rec_params, new_rec_state), (new_gen_params, new_gen_state), new_opt_state), (loss_val, nll, kl)


def sample_data(
    rng_key: PRNGKey,
    gen_params: ArrayLikeTree,
    gen_state: ArrayLikeTree,
    gen_apply_fn: Callable,
    n_samples: int,
    latent_dim: int,
):
    z = jax.random.normal(rng_key, shape=(n_samples, latent_dim))
    x, gen_state = gen_apply_fn(params=gen_params, state=gen_state, input=z, train=False)
    return x, gen_state




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


def AEVB(
    latent_dim: int,
    generative_apply: Callable,
    recognition_apply: Callable,
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

    gen_apply = generative_apply
    rec_apply = recognition_apply

    def init_fn(rec_params, rec_state, gen_params, gen_state) -> AEVBState:
        opt_state = optimizer.init((rec_params, gen_params))
        return AEVBState(rec_params, rec_state, gen_params, gen_state, opt_state)

    @jit
    def step_fn(rng_key, aevb_state, x) -> tuple[AEVBState, AEVBInfo]:
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
        ((new_rec_params,new_rec_state), (new_gen_params, new_gen_state), new_opt_state), (loss_val, nll, kl) = (
            tractable_kl_step(
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
        )
        return AEVBState(new_rec_params, new_rec_state, new_gen_params, new_gen_state, new_opt_state), AEVBInfo(
            loss_val, nll, kl
        )

    @bind(jit, static_argnames=["n_samples"])
    def sample_data_fn(rng_key, gen_params, gen_state, n_samples) -> ArrayLike:
        """_summary_

        Args:
            rng_key (_type_): _description_
            gen_params (_type_): _description_
            n_samples (_type_): _description_

        Returns:
            ArrayLike: _description_
        """
        return sample_data(
            rng_key, gen_params, gen_state, gen_apply, n_samples, latent_dim
        )

    return init_fn, step_fn, sample_data_fn
