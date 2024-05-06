import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import optax
from datasets import load_dataset

from aevb.aevb import Aevb, AevbEngine, AevbState
from aevb.equinox_util import init_apply_eqx_model
from aevb.util import decode, encode, generate_random_samples


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


@eqx.nn.make_with_state
class Encoder(eqx.Module):

    latent_dim: int
    layers: list
    projection_layers: list

    def __init__(self, key: random.key, latent_dim: int):
        keys = random.split(key, 6)
        self.latent_dim = latent_dim

        self.layers = [
            eqx.nn.Linear(in_features=784, out_features=512, key=keys[0]),
            eqx.nn.BatchNorm(input_size=512, axis_name=("batch", "batch2")),
            jax.nn.relu,
            eqx.nn.Linear(in_features=512, out_features=256, key=keys[1]),
            jax.nn.relu,
            eqx.nn.Linear(in_features=256, out_features=128, key=keys[2]),
            jax.nn.relu,
            eqx.nn.Linear(in_features=128, out_features=64, key=keys[3]),
        ]
        self.projection_layers = [
            eqx.nn.Linear(in_features=64, out_features=self.latent_dim, key=keys[4]),
            eqx.nn.Linear(in_features=64, out_features=self.latent_dim, key=keys[5]),
        ]

    def __call__(self, x, state):
        for layer in self.layers:
            if isinstance(layer, eqx.nn._batch_norm.BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)

        mu = self.projection_layers[0](x)
        logvar = self.projection_layers[1](x)
        sigma = jnp.exp(logvar * 0.5)
        return {"loc": mu, "scale": sigma}, state


@eqx.nn.make_with_state
class Decoder(eqx.Module):

    latent_dim: int
    layers: list

    def __init__(self, key: random.key, latent_dim: int):
        keys = random.split(key, 3)
        self.latent_dim = latent_dim
        self.layers = [
            eqx.nn.Linear(in_features=self.latent_dim, out_features=128, key=keys[0]),
            eqx.nn.BatchNorm(input_size=128, axis_name=("batch", "batch2")),
            jax.nn.relu,
            eqx.nn.Linear(in_features=128, out_features=256, key=keys[1]),
            jax.nn.relu,
            eqx.nn.Linear(in_features=256, out_features=784, key=keys[2]),
        ]

    def __call__(self, x, state):
        for layer in self.layers:
            if isinstance(layer, eqx.nn._batch_norm.BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)
        return {"loc": x, "scale": jnp.ones_like(x) * 0.1}, state


# Main Function --------------------------------
def main(save_samples_pth: str):

    # Prepare Data
    mnist_data = load_dataset("mnist")
    data_train = mnist_data["train"]

    X_train = np.stack([np.array(example["image"]) for example in data_train])
    X_train, N_train = prepare_data(X_train)

    seed = 1
    n = N_train.item()
    batch_size = 100
    batches = data_stream(seed, X_train, batch_size, n)

    # Create AEVB inference engine
    decoder = Decoder(random.key(0), latent_dim=4)
    encoder = Encoder(random.key(1), latent_dim=4)
    latent_dim = 4
    data_dim = 784
    optimizer = optax.adam(1e-3)

    dec_init, dec_apply = init_apply_eqx_model(decoder, batchnorm=True, input_dim=latent_dim)
    enc_init, enc_apply = init_apply_eqx_model(encoder, batchnorm=True, input_dim=data_dim)

    engine: AevbEngine = Aevb(
        latent_dim=latent_dim,
        data_dim=data_dim,
        enc_apply=enc_apply,
        enc_init=enc_init,
        dec_apply=dec_apply,
        dec_init=dec_init,
        prior="unit_normal",
        obs_dist="normal",
        variational_dist="normal",
        optimizer=optimizer,
        n_samples=5,
    )

    # Run AEVB
    key = random.key(1242)
    num_steps = 3000
    eval_every = 100

    aevb_state: AevbState = engine.init(dec_init_args=(), enc_init_args=())

    key, *training_keys = random.split(key, num_steps + 1)
    for i, rng_key in enumerate(training_keys):
        batch = next(batches)
        aevb_state, info = jax.jit(engine.step)(rng_key, aevb_state, batch)
        if i % eval_every == 0:
            print(f"Step {i} | loss: {info.loss} | nll: {info.nll} | kl: {info.kl}")

    key, data_samples_key = random.split(key)
    x_samples = generate_random_samples(
        data_samples_key, engine, aevb_state, n_samples=5
    )

    key, encode_key, decode_key = random.split(key, 3)
    z_samples = encode(encode_key, engine, aevb_state, x_samples, n_samples=30)
    z_means = z_samples.mean(axis=0)
    x_recon = decode(decode_key, engine, aevb_state, z_means)

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
