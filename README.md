**DISCLAIMER**: work in progress...

# Autoencoding Variational Bayes
Applying the AEVB Estimator to Latent Variable Models

## Overview

The purpose of this package is to provide a simple (but general) implementation of the Auto-Encoding Variational Bayes (AEVB) inference algorithm ([Kingma et. al, 2014](https://arxiv.org/abs/1312.6114)) as well as a composable and interoperable interface for the implementation.

### Interoperability
- [x] Arbitrary `init/apply` callables for the encoder/decoder mappings.
- [x] Flax modules for the initialization and encoder/decoder mappings.
- [x] Equinox modules for the initialization and encoder/decoder mappings.
- [ ] Haiku modules for the initialization and encoder/decoder mappings.

### Gradient Estimator Support
- [x] Reparameterization trick support for loc/scale distribution families.
- [ ] Reparameterization trick support for tractable inverse CDF distribution families.
- [ ] Reparameterization trick support for composable distribution families.
- [ ] REINFORCE estimator support.

### KL Support
- [x] Support for analytical KL term for normal prior and normal recognition model distribution. 
- [ ] Support for analytical KL term for other prior and recognition model distributions.
- [ ] Support for intractable KL term.


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

1. `latent_dim: int`: the dimension of the latent variable $z$. 
2. `data_dim: int`: the dimension of the data $x$. 
3. `gen_prior: str | Callable`: the logpdf function of a prior distribution over continuous latent variable $z$. 
4. `gen_obs_dist: str | Callable`: the logpdf function of a distribution over the data $x$.
6. `gen_apply`: a function mapping learnable parameters and latent variable $z$ to the parameters of the `obs_dist`.
7. `gen_init`: an initialization for the function mapping defined by `gen_apply`. 
8. `rec_dist`: the logpdf function and reparameterized sample function of a distribution over continuous latent variable $z$. 
9. `rec_apply`: a function mapping learnable parameters and data variable $x$ to the parameters of the `rec_dist`.
10. `rec_init`: an initialization for the function mapping defined by `rec_apply`. 
11. `optimizer`: an `optax` optimizer.
12. `n_samples`: the number of samples to take from the reparameterized sample function of `rec_dist` during one step of optimization. 

### Restrictions on `apply`
The `gen_apply` and `rec_apply` callables need to have a specific signature and form in order to work with `aevb`:

The signature can be represented like so:
```python
def apply(params: ArrayLikeTree, state: ArrayLikeTree, input: ArrayLike, train: bool): -> tuple[Dict[str, Array], ArrayLikeTree]
```

The `apply` function should apply parameters and a (optional) state to an input in order to produce a tuple consisting of the output and a (optional) updated state. The `train` boolean flag is for the purpose of possibly updating the state. This is a common pattern in the `flax` library. An example of a `state` is the current values for the Batchnorm statistics. If no state is needed, then one can bind an empty dictionary to `state` as well as to the second element of the output tuple. 

Another restriction on `apply` is that the first element of the output tuple is a dictionary mapping string keys to `jax` Arrays. *These keys need to correspond to the* `dist` *of the encoder/decoder*. That is, for the `gen_apply`, the keys need to correspond to the input arguments of `gen_obs_dist` logpdf callable.


### Restrictions on `init`
The `gen_init` and `rec_init` callables need only have a specific output form in order to work with `aevb`. That is, they both need to return a tuple of `params, state`. Again, if the `state` is not needed this can be an empty dictionary `{}`. 

### Using Flax Modules for Encoder/Decoder
`aevb` provides functions that convert `flax` modules into `init/apply` functions that conform with the above specifications. The only restriction is that the `__call__` method of the `flax` module has a `train: bool` input argument. It is not required that it used in the method, however. Here is a simple example.

```python
import jax
import jax.numpy as jnp
import flax.linen as nn

class Mlp(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = False):
        x = nn.Dense(14, x)(x)


```

### Using Equinox Modules for Encoder/Decoder

1.  `__call__` should only have `x, state` as arguments, thus all layer instantiation should be done in `__init__` since they require a random key for initialization.


### Using Haiku Modules for Encoder/Decoder



## Installation

### User Install
By default, installing the package will install the CPU version of `jax`. If you have a GPU and would like to utilize it, first install the [`jax` build](https://jax.readthedocs.io/en/latest/installation.html) that fits your hardware specifications. 

Also, if you want to use a supported neural network library (e.g. Flax), you can append its name to the commands below.

| Description | Command |
----------| ---------| 
| Minimal install | `pip install git+https://github.com/PaulScemama/aevb`|


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

