# Autoencoding Variational Bayes
Applying the AEVB Estimator to Latent Variable Models

## Overview

The purpose of this package is to provide a simple (but general) implementation of the Auto-Encoding Variational Bayes (AEVB) inference algorithm ([Kingma et. al, 2014](https://arxiv.org/abs/1312.6114)) as well as a composable and interoperable interface for the implementation.

### Interoperability
- [x] Arbitrary `apply` callables for the encoder/decoder mappings.
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

1. `latent_dim`: the dimension of the latent variable $z$. 
2. `data_dim`: the dimension of the data $x$. 
3. A `gen_prior`: the logpdf function of a prior distribution over continuous latent variable $z$. 
4. A `gen_obs_dist`: the logpdf function of a distribution over the data $x$. 
5. A `gen_apply`: a function mapping learnable parameters and latent variable $z$ to the parameters of the `obs_dist`. 
6. A `rec_dist`: the logpdf function and reparameterized sample function of a distribution over continuous latent variable $z$. 
7. A `rec_apply`: a function mapping learnable parameters and data variable $x$ to the parameters of the `rec_dist`. 
8. An `optimizer`: an `optax` optimizer.
9. `n_samples`: the number of samples to take from the reparameterized sample function of `rec_dist` during one step of optimization. 

An example using `jax` functions for `gen_apply` and `rec_apply` are given in `examples/jax/`.

> Note: there is builtin support for distributions. For example, one can pass in the string `'unit_normal'` as `gen_prior`; or one can pass in the string `'normal'` as `gen_obs_dist` or `rec_dist`. 

The signature of `gen_apply` and `rec_apply` must be the following: 

```python
(params: ArrayLikeTree, state: ArrayLikeTree, input: ArrayLike, train: bool) -> Dict[str, ArrayLike]
```

In words, the `apply` methods take in learnable parameters, a learnable state (if none is required, pass in an empty dictionary `{}`), input data, and a boolean flag indicating whether this is being used during training of the learnable components passed in. The output of `gen_apply` and `rec_apply` must output dictionaries where the keys are the keyword arguments for the `gen_obs_dist` logpdf and `rec_dist` logpdf and reparameterized sample, respectively. 

When the components listed above are passed to the `AevbEngine` contructor, two main functions are then available for the user.

1. An `init` function. This takes in the learnable parameters (and possibly learnable state) and returns an `AevbState` object, which contains the learnable parameters (and possibly learnable state) and an optimization state.
2. A `step` function. This takes in a `random.key`, an `AEVBState`, and a batch of data `x`, and returns an updated `AEVBState` after one step of optimization. 

Other objects are available to the user from the `AevbEngine` as well. Look at `aevb._src.aevb.AevbEngine` for more information. 

While the above 'user inputs' list contains the *required* inputs needed to create an `aevb` inference engine, a user may want to pass in higher-level objects (e.g. Flax or Equinox modules). The following sections demonstrate how to do this, and what changes about the properties of the `aevb` inference engine. 

### Using Flax Modules for Encoder/Decoder

You can pass in `flax` modules in place of `gen_apply` and `rec_apply`. Why would you want to do this? You can initialize the first `AevbState` with `AevbEngine.init(key: random.key)` instead of having to initialize the parameters and state for the encoder/decoder and then pass that into `AevbEngine.init`. You also don't need to explicitly create the `apply` functions of the encoder/decoder modules. 

Here's an example of what this may look like

```python
import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn

class Encoder(nn.Module):

    @nn.compact
    def __call__(self, x, train: bool = False):
        x = nn.Dense(5)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        # rec_dist is of the loc/scale family. Here we fix
        # the `scale` parameter and only learn how to construct `loc`. 
        return {"loc": x, "scale": jnp.ones_like(x)}

class Decoder(nn.Module):
    
    @nn.compact
    def __call__(self, x, train: bool = False)
        x = nn.Dense(25)(x)
        x = nn.relu(x)
        # gen_obs_dist is of the loc/scale family. Here we fix
        # the `scale` parameter and only learn how to construct `loc`. 
        return {"loc": x, "scale": jnp.ones_like(x)} 

encoder = Encoder()
decoder = Decoder()

engine = AevbEngine.from_flax_modules(
    ...
    gen_module=decoder,
    rec_moodule=encoder,
)

aevb_state = engine.init(random.key(1))
```

### Using Equinox Modules for Encoder/Decoder


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

