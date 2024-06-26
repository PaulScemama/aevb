**DISCLAIMER**: work in progress...

# Autoencoding Variational Bayes
Applying the AEVB Estimator to Latent Variable Models

## Table of Contents
1. [Overview](#overview)
    1. [Interoperability](#interoperability)
    2. [Distribution Support](#distribution-support)
    3. [KL Support](#kl-support)
2. [Auto-Encoding Variational Bayes (AEVB)](#auto-encoding-variational-bayes-aevb)
3. [How to Use](#how-to-use)
    1. [Restrictions on `apply`](#restrictions-on-apply)
    2. [Restrictions on `init`](#restrictions-on-init)
    3. [Using Flax Modules for Encoder/Decoder](#using-flax-modules-for-encoderdecoder)
    4. [Using Equinox Modules for Encoder/Decoder](#using-equinox-modules-for-encoderdecoder)
    5. [Using Haiku Modules for Encoder/Decoder](#using-haiku-modules-for-encoderdecoder)
4. [User Install](#user-install)
5. [Developer Install](#developer-install)

## Overview

The purpose of this package is to provide a simple (but general) implementation of the Auto-Encoding Variational Bayes (AEVB) inference algorithm ([Kingma et. al, 2014](https://arxiv.org/abs/1312.6114)) as well as a composable and interoperable interface for the implementation.

### Interoperability
- [x] Arbitrary `init/apply` callables for the encoder/decoder mappings.
- [x] Flax modules for the initialization and encoder/decoder mappings.
- [x] Equinox modules for the initialization and encoder/decoder mappings.
- [ ] Haiku modules for the initialization and encoder/decoder mappings.

### Distribution Support
- [x] Reparameterization trick support for loc/scale distribution families.
- [ ] Reparameterization trick support for tractable inverse CDF distribution families.
- [ ] Reparameterization trick support for composable distribution families.
- [ ] REINFORCE estimator support.

### KL Support
- [x] Support for analytical KL term for normal prior and normal recognition model distribution. 
- [x] Support for intractable KL term.
- [ ] Support for analytical KL term for other prior and recognition model distributions.



## Auto-Encoding Variational Bayes (AEVB)

I think of AEVB working with a generative model and a recognition model. The generative model consists of a prior over continuous latent variables $z$, a mapping from latent variables $z$ to *the parameters of a distribution* over observed data variable $x$, and the actual form of the "observation distribution" over $x$. In other words,

$$
\text{generative model} = p(x,z) = p(z)p(x|g_{\theta}(z))
$$

I often refer to $g_{\theta}$ as the "decoder". 

The recognition model consists of a mapping from data $x$ to *the parameters of a distribution* over continuous latent variables $z$, and the actual form of the distribution over $z$. In other words,

$$
\text{recognition model} = q(z|f_{\phi}(x))
$$

I often refer to $f_{\phi}$ as the "encoder". 


## How To Use

In order to use `aevb`, the user must define...
| Parameter name | Type | Description | 
| ----------------| -------------- | --------------- |
| `latent_dim` | `int` | The dimension of the latent variable $z$.|
|`data_dim` | `int` | The dimension of the data $x$.| 
| `gen_prior`| `str \| Callable` | The logpdf function of a prior distribution over continuous latent variable $z$, or a string corresponding to a built-in prior.|
|`gen_obs_dist`| `str \| Callable`| The logpdf function of a distribution over the data $x$, or a string corresponding to a built-in distribution.|
| `gen_apply` |`Callable`|A function mapping learnable parameters and latent variable $z$ to the parameters of the `obs_dist`.|
| `gen_init`| `Callable`| An initialization for the parameters and state that will be passed into `gen_apply`.| 
|`rec_dist`| `str \| tuple[Callable, Callable]`| The logpdf function and reparameterized sample function of a distribution over continuous latent variable $z$, or a string corresopnding to a built-in distribution.|
|`rec_apply`|  `Callable` |A function mapping learnable parameters and data variable $x$ to the parameters of the `rec_dist`.|
|`rec_init` |`Callable`| An initialization for the parameters and state that will be passed into `rec_apply`.| 
|`optimizer`| `GradientTransformation`| An `optax` optimizer.|
|`n_samples`| `int`| The number of samples to take from the reparameterized sample function of `rec_dist` during one step of optimization. |

### Restrictions on `apply`
The `gen_apply` and `rec_apply` callables need to have a specific signature and form in order to work with `aevb`:

The signature can be represented like so:

```python
import jax
from jax.typing import ArrayLike

ArrayTree = jax.Array | Iterable["ArrayTree"] | Mapping[Any, "ArrayTree"]
ArrayLikeTree = ArrayLike | Iterable["ArrayLikeTree"] | Mapping[Any, "ArrayLikeTree"]

def apply(
    params: ArrayLikeTree, 
    state: ArrayLikeTree, 
    input: ArrayLike, 
    train: bool
): -> tuple[Dict[str, Array], ArrayLikeTree]
    ...
    return out, state

```

- The `apply` function should apply parameters and a (optional) state to an input in order to produce a tuple consisting of the output and an (optional) updated state. The `train` boolean flag is for the purpose of possibly updating the state. This is a common pattern in the `flax` library. 
- An example of a `state` is a representation ofthe current values for the Batchnorm statistics. 
- If no state is needed, then one can bind an empty dictionary to `state` as well as to the second element of the output tuple. 

Another restriction on `apply` is that the first element of the output tuple is a dictionary mapping string keys to `jax` Arrays. *These keys need to correspond to the* `*_dist` *of the encoder/decoder*. 
- That is, for the `gen_apply`, the keys need to correspond to the input arguments of `gen_obs_dist` logpdf callable.


### Restrictions on `init`
The `gen_init` and `rec_init` callables need only have a specific output form in order to work with `aevb`. That is, they both need to return a tuple consisting of `params, state`. Similarly, if the `state` is not needed this can be an empty dictionary `{}`. 

### Using Flax Modules for Encoder/Decoder
`aevb` provides functions that convert `flax` modules into `init/apply` functions that conform with the above specifications. The only restriction is that the `__call__` method of the `flax` module has a `train: bool` input argument. It is not required that this `train` arg is actually used in the method, however. Here is a simple example.

```python
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from aevb.flax_util import init_apply_flax_model

class Mlp(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = False):
        x = nn.Dense(7)(x)
        x = nn.Dense(1)(x)
        return {"x": x}

mlp = Mlp()
init, apply = init_apply_flax_model(mlp)
params, state = init(random.key(0), jnp.ones(6))
apply(params, state, jnp.ones((10, 6)), train=True)
# ({'x': Array([[0.9643452],
#          [0.9643452],
#          [0.9643452],
#          [0.9643452],
#          [0.9643452],
#          [0.9643452],
#          [0.9643452],
#          [0.9643452],
#          [0.9643452],
#          [0.9643452]], dtype=float32)},
#  {})
```
- Note how when converting a `flax.nn.Module` to `init/apply`, the `init` requires a random key and an array with the same shape as the data. 

### Using Equinox Modules for Encoder/Decoder
`aevb` provides functions that convert `equinox` modules into `init/apply` functions that conform with our specifications. The only restrictions are:

1. The `__call__` method should only have `x, state` as arguments, thus all layer instantiation should be done in `__init__`. Here is a simple example.
2. The module should be transformed by `equinox.nn.make_with_state`. This can conveniently done with a decorator (see following example)

Here is a simple example.

```python
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
from aevb.equinox_util import init_apply_eqx_model

@eqx.nn.make_with_state
class Mlp(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, key: random.key):
        l1_key, l2_key = random.split(key)
        self.l1 = eqx.nn.Linear(in_features=6, out_features=7, key=l1_key)
        self.l2 = eqx.nn.Linear(in_features=7, out_features=1, key=l2_key)

    def __call__(self, x, state):
        x = self.l1(x)
        x = self.l2(x)
        return {"x": x}, state

model = Mlp(random.key(0))
init, apply = init_apply_eqx_model(model, batchnorm=False, input_dim=6)
params, state = init()
apply(params, state, jnp.ones((10, 6)), train=False)
# ({'x': Array([[0.3000745],
#          [0.3000745],
#          [0.3000745],
#          [0.3000745],
#          [0.3000745],
#          [0.3000745],
#          [0.3000745],
#          [0.3000745],
#          [0.3000745],
#          [0.3000745]], dtype=float32)},
#  State())
```

- Note how when converting from an `equinox.Module`, the `init` takes no input arguments.

### Using Haiku Modules for Encoder/Decoder

Work in progress...


## Installation

### User Install
By default, installing the package will install the CPU version of `jax`. If you have a GPU and would like to utilize it, first install the [`jax` build](https://jax.readthedocs.io/en/latest/installation.html) that fits your hardware specifications. 

Also, if you want to use a supported neural network library (e.g. Flax), you can append its name to the commands below.

| Description | Command |
----------| ---------| 
| Minimal install | `pip install git+https://github.com/PaulScemama/aevb`|

> Note: to run an example you will have to install other packages. For example, to run `examples/flax/mnist.py` you will need to install `datasets`, `matplotlib`, and `flax`. 
<!-- |Install with example dependencies| `pip install 'git+https://github.com/PaulScemama/aevb[examples]'`| -->


### Developer Install
Clone the repository; and within the root of the project, create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install project in editable mode along with its dependencies.

```bash
pip install -e .[examples]
```

