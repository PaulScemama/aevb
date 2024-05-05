import inspect
from typing import Any, Callable, Dict, List, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
from jax.random import PRNGKey

State = eqx.nn._stateful.State


def _maybe_modify_input_ndim(input, input_dim):
    if isinstance(input_dim, int):
        data_ndim = 1
    else:
        data_ndim = len(input_dim)
    # input has [n_samples, batch, ...] we're good.
    if input.ndim - data_ndim == 2:
        return input, False
    # input has [batch, ...]
    elif input.ndim - data_ndim == 1:
        return jnp.expand_dims(input, axis=0), True
    else:
        raise ValueError("Input to `apply` function from equinox must either have shape [n_samples, batch, ...] or [batch, ...]")
    

def batch_model(model: eqx.Module, batchnorm: bool) -> Callable:
    if batchnorm:
        # see BatchNorm: https://docs.kidger.site/equinox/api/nn/normalisation/
        # Across typical batch dimension
        first_vmap = jax.vmap(model, in_axes=(0, None), out_axes=(0, None), axis_name="batch")
        # Across n_samples dimension
        second_vmap = jax.vmap(first_vmap, in_axes=(0, None), out_axes=(0, None), axis_name="batch2")
        return second_vmap
    else:
        # Across typical batch dimension
        first_vmap = jax.vmap(model, in_axes=(0, None), out_axes=(0, None))
        # Across n_samples dimension
        second_vmap = jax.vmap(first_vmap, in_axes=(0, None), out_axes=(0, None))
        return second_vmap


def init_apply_eqx_model(
    model: tuple[Any, State], batchnorm: bool, input_dim: Union[int, tuple[int, ...]]
) -> tuple[Callable, Callable]:
    model, state = model
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def init():
        return params, state

    def apply(params, state, input, train: bool):
        input, squeeze = _maybe_modify_input_ndim(input, input_dim)
        model = eqx.combine(params, static)
        batched_model = batch_model(model, batchnorm)
        if not train:
            model = eqx.nn.inference_mode(model)
        out, updates = batched_model(input, state)
        if squeeze:
            out = {k: jnp.squeeze(v, axis=0) for k,v in out.items()}
        return out, updates
    return init, apply


@eqx.nn.make_with_state
class MLP(eqx.Module):

    in_dim: int
    layers: List
    output_heads: List[eqx.nn.Linear]

    def __init__(
        self,
        key: random.key,
        in_dim: int,
        hidden: List[int],
        activation: tuple[Callable, List[int]],
        batchnorm_idx: List[int],
        output_heads: Dict[str, Union[int, tuple[int, callable]]],
    ):

        keys = random.split(key, len(hidden) + 1)
        act_fn: Callable = activation[0]
        act_idx: List[int] = activation[1]

        layers = []
        # Current position in [latent_dim, hidden[0], ..., out_dim]
        # meaning that x right now is of shape latent_dim (the 0th idx).
        i = 0

        def add_block(idx: int):
            if idx == 0:
                # Mapping from input layer to hidden layer.
                layers.append(eqx.nn.Linear(in_dim, hidden[idx], key=keys[idx]))
            else:
                # Mapping from hidden layer to hidden layer.
                layers.append(
                    eqx.nn.Linear(hidden[idx - 1], hidden[idx], key=keys[idx])
                )

            if idx in batchnorm_idx:
                batchnorm_layer = eqx.nn.BatchNorm(
                    input_size=hidden[idx], axis_name=("batch", "batch2")
                )
                layers.append(batchnorm_layer)
            if idx in act_idx:
                layers.append(act_fn)

        while i < len(hidden):
            add_block(i)
            i += 1

        self.layers = layers

        self.output_heads = {}
        self.in_dim = in_dim
        for name, shape_and_transform in output_heads.items():
            if isinstance(shape_and_transform, tuple):
                shape, transform = shape_and_transform
                projection_layer = eqx.nn.Linear(hidden[-1], shape, key=keys[i])
                self.output_heads[name] = (projection_layer, transform)
            else:
                shape = shape_and_transform
                projection_layer = eqx.nn.Linear(hidden[-1], shape, key=keys[i])
                self.output_heads[name] = projection_layer

    def __call__(self, x, state):
        for layer in self.layers:
            if isinstance(layer, eqx.nn._batch_norm.BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)

        out = {}
        for name, layer_and_transform in self.output_heads.items():
            if isinstance(layer_and_transform, tuple):
                layer, transform = layer_and_transform
                out[name] = transform(layer(x))
            else:
                layer = layer_and_transform
                out[name] = layer(x)

        return out, state
