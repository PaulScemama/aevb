import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import optax
from datasets import load_dataset
from jax.random import PRNGKey, split

from aevb.core import AEVB


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
class RecModel(eqx.Module):

    latent_dim: int
    layers: list
    projection_layers: list

    def __init__(self, key: PRNGKey, latent_dim: int):
        keys = random.split(key, 6)
        self.latent_dim = latent_dim

        self.layers = [
            eqx.nn.Linear(in_features=784, out_features=512, key=keys[0]),
            eqx.nn.BatchNorm(input_size=512, axis_name="batch"),
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
        return (mu, sigma), state


@eqx.nn.make_with_state
class GenModel(eqx.Module):

    latent_dim: int
    layers: list

    def __init__(self, key: PRNGKey, latent_dim: int):
        keys = random.split(key, 3)
        self.latent_dim = latent_dim
        self.layers = [
            eqx.nn.Linear(in_features=self.latent_dim, out_features=128, key=keys[0]),
            eqx.nn.BatchNorm(input_size=128, axis_name="batch"),
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
        return x, state


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
    gen_model = GenModel(random.key(0), latent_dim=4)
    rec_model = RecModel(random.key(1), latent_dim=4)
    latent_dim = 4
    optimizer = optax.adam(1e-3)
    engine = AEVB(
        latent_dim=latent_dim,
        generative_model=gen_model,
        recognition_model=rec_model,
        optimizer=optimizer,
        n_samples=15,
        nn_lib="equinox",
    )

    # Run AEVB
    key = PRNGKey(1242)
    num_steps = 1000
    eval_every = 100

    aevb_state = engine.init()

    key, *training_keys = split(key, num_steps + 1)
    for i, rng_key in enumerate(training_keys):
        batch = next(batches)
        aevb_state, info = engine.step(rng_key, aevb_state, batch)
        if i % eval_every == 0:
            print(f"Step {i} | loss: {info.loss} | nll: {info.nll} | kl: {info.kl}")

    # Random Data Samples of Learned Generative Model
    key, data_samples_key = split(key)
    x_samples = engine.util.sample_data(
        data_samples_key, aevb_state.gen_params, aevb_state.gen_state, 5
    )
    fig, axs = plt.subplots(5, 1)
    for i, s in enumerate(x_samples):
        axs[i].imshow(s.reshape(28, 28))
    plt.savefig(save_samples_pth, format="png")

    # Encode samples
    z_samples = engine.util.encode(key, aevb_state, x_samples, n_samples=13)
    print(z_samples.shape)


if __name__ == "__main__":
    from time import localtime, strftime

    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    main(f"./samples-{now}.png")
