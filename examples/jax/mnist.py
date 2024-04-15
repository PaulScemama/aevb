import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import optax
from datasets import load_dataset

from aevb._src.types import ArrayLike
from aevb.aevb import Aevb, AevbEngine, AevbState


# Data Processing Functions ----------------------------------
def one_hot_encode(x, k):
    "Create a one-hot encoding of x of size k."
    return jnp.array(x[:, None] == jnp.arange(k), dtype=jnp.float32)


@jax.jit
def prepare_data(X):
    num_examples = X.shape[0]
    num_pixels = 28 * 28
    X = X.reshape(num_examples, num_pixels)
    X = X / 255.0

    return X, num_examples


def data_stream(seed, data, batch_size, data_size):
    """Return an iterator over batches of data."""
    rng = np.random.RandomState(seed)
    num_batches = int(jnp.ceil(data_size / batch_size))
    while True:
        perm = rng.permutation(data_size)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            yield data[batch_idx]


# Generative Model and Recognition Feature Extractor --------------------
def recognition_init(key, latent_dim, data_dim):
    w1key, b1key, w2key, b2key, w3key, b3key = random.split(key, 6)

    shared_W = random.normal(w1key, (data_dim, 100)) * 0.1
    shared_b = random.normal(b1key, (100,)) * 0.1

    mu_W = random.normal(w2key, (100, latent_dim)) * 0.1
    mu_b = random.normal(b2key, (latent_dim,)) * 0.1

    logvar_W = random.normal(w3key, (100, latent_dim)) * 0.1
    logvar_b = random.normal(b3key, (latent_dim,)) * 0.1

    return (
        {
            "shared": {"W": shared_W, "b": shared_b},
            "mu": {"W": mu_W, "b": mu_b},
            "logvar": {"W": logvar_W, "b": logvar_b},
        }
    ), {}


def recognition_apply(params, state, input, train):
    x = jnp.dot(input, params["shared"]["W"]) + params["shared"]["b"]
    x = jax.nn.relu(x)
    mu = jnp.dot(x, params["mu"]["W"]) + params["mu"]["b"]
    logvar = jnp.dot(x, params["logvar"]["W"]) + params["logvar"]["b"]
    sigma = jnp.exp(logvar * 0.5)
    return {"loc": mu, "scale": sigma}, {}


def generative_init(key, latent_dim, data_dim):
    wkey, bkey = random.split(key)
    W = random.normal(wkey, (latent_dim, data_dim)) * 0.1
    b = random.normal(bkey, (data_dim,)) * 0.1
    return {"w": W, "b": b}, {}


def generative_apply(params, state, input, train: bool):
    W, b = params["w"], params["b"]
    pre = jnp.dot(input, W) + b
    out = jax.nn.relu(pre)
    return {"loc": out, "scale": 0.1}, {}


# Main Function --------------------------------
def main(save_samples_pth: str):
    # Prepare Data
    mnist_data = load_dataset("mnist")
    data_train = mnist_data["train"]

    X_train = np.stack([np.array(example["image"]) for example in data_train])
    X_train, N_train = prepare_data(X_train)

    seed = 1
    n = N_train.item()
    batch_size = 500
    batches = data_stream(seed, X_train, batch_size, n)

    data_dim = 784
    latent_dim = 4
    optimizer = optax.adam(1e-3)

    engine: AevbEngine = Aevb(
        latent_dim=latent_dim,
        data_dim=data_dim,
        gen_prior="unit_normal",
        gen_obs_dist="normal",
        gen_apply=generative_apply,
        gen_init=generative_init,
        rec_dist="normal",
        rec_apply=recognition_apply,
        rec_init=recognition_init,
        optimizer=optimizer,
        n_samples=1,
    )

    # Run AEVB
    key = random.key(1242)
    num_steps = 5000
    eval_every = 100

    key, subkey1, subkey2 = random.split(key, 3)
    aevb_state = engine.init(
        gen_init_args=(subkey1, latent_dim, data_dim),
        rec_init_args=(subkey2, latent_dim, data_dim),
    )

    key, *training_keys = random.split(key, num_steps + 1)
    for i, rng_key in enumerate(training_keys):
        batch = next(batches)
        aevb_state, info = jax.jit(engine.step)(rng_key, aevb_state, batch)
        if i % eval_every == 0:
            print(f"Step {i} | loss: {info.loss} | nll: {info.nll} | kl: {info.kl}")

    # Random Data Samples of Learned Generative Model
    def generate_random_samples(
        key: random.key, aevb_engine: AevbEngine, aevb_state: AevbState, n_samples: int
    ):
        z_key, x_key = random.split(key)
        if aevb_engine.gen_model.prior.sample is not None:
            prior_zs = aevb_engine.gen_model.prior.sample(
                z_key, shape=(n_samples, latent_dim)
            )
        else:
            # defer to N(0, 1)
            prior_zs = jax.random.normal(
                z_key, shape=(n_samples, aevb_engine.latent_dim)
            )

        x_params, _ = aevb_engine.gen_model.apply(
            aevb_state.gen_params, aevb_state.gen_state, prior_zs, train=False
        )

        if aevb_engine.gen_model.obs_dist.sample is not None:
            xs = aevb_engine.gen_model.obs_dist.sample(x_key, **x_params, shape=())
            return xs
        else:
            return x_params

    key, data_samples_key = random.split(key)
    x_samples = generate_random_samples(
        data_samples_key, engine, aevb_state, n_samples=5
    )

    # Encode/Decode samples using Learned Recognition and Generative Models
    def encode(
        key: random.key,
        aevb_engine: AevbEngine,
        aevb_state: AevbState,
        x: ArrayLike,
        n_samples: int,
    ):
        rec_model = aevb_engine.rec_model
        z_params, _ = rec_model.apply(
            aevb_state.rec_params, aevb_state.rec_state, x, train=False
        )
        return rec_model.dist.reparam_sample(key, **z_params, n_samples=n_samples)

    def decode(
        key: random.key,
        aevb_engine: AevbEngine,
        aevb_state: AevbEngine,
        z: ArrayLike,
        n_samples: int,
    ):
        x_params, _ = aevb_engine.gen_model.apply(
            aevb_state.gen_params, aevb_state.gen_state, z, train=False
        )
        if aevb_engine.gen_model.obs_dist.sample is not None:
            xs = aevb_engine.gen_model.obs_dist.sample(
                key, **x_params, shape=(n_samples,)
            )
            return xs
        else:
            return x_params

    key, encode_key, decode_key = random.split(key, 3)
    z_samples = encode(encode_key, engine, aevb_state, x_samples, n_samples=30)
    z_means = z_samples.mean(axis=0)
    x_recon = decode(decode_key, engine, aevb_state, z_means, n_samples=1)

    fig, axs = plt.subplots(2, 5)
    for i, s in enumerate(x_samples):
        axs[0][i].imshow(s.reshape(28, 28))
    for i, s in enumerate(x_recon):
        axs[1][i].imshow(s.reshape(28, 28))

    plt.figtext(
        0.5, 0.95, "Random Generative Samples", ha="center", va="top", fontsize=14
    )
    plt.figtext(0.5, 0.5, "Their Reconstructions", ha="center", va="top", fontsize=14)
    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()

    plt.savefig(save_samples_pth, format="png")


if __name__ == "__main__":
    from time import localtime, strftime

    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    main(f"./samples-{now}.png")
