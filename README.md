# Autoencoding Variational Bayes
Applying the AEVB Estimator to Latent Variable Models

## Overview

The purpose of this package is to provide a simple (but general) implementation of the Auto-Encoding Variational Bayes (AEVB) inference algorithm ([Kingma et. al, 2014](https://arxiv.org/abs/1312.6114)) as well as a composable and interoperable interface for the implementation.


- [x] Arbitrary `apply` callables for the encoder/decoder mappings.
- [ ] Arbitrary `init` and `apply` callables for the initialization and encoder/decoder mappings.
- [x] Flax modules for the initialization and encoder/decoder mappings.
- [x] Equinox modules for the initialization and encoder/decoder mappings.
- [ ] Haiku modules for the initialization encoder/decoder mappings.
- [x] Reparameterization trick support for loc/scale families.
- [ ] Reparameterization trick support for tractable inverse CDF families.
- [ ] Reparameterization trick support for composable families.
- [ ] REINFORCE estimator support.
- [x] Support for analytical KL term for normal/normal prior and recognition model distribution. 
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

