from dataclasses import dataclass
from typing import Dict, List


@dataclass
class JaxConfig:
    gpus: List[int]


@dataclass
class RunConfig:
    update_steps: int
    eval_every: int


@dataclass
class ModelsConfig:
    latent_dim: int
    data_shape: List[int]

    gen_type: str
    gen_hidden: List[int]
    # For *_norm and *_act, the List[int] argument is a list of
    # indices which indicate an application AFTER the corresponding
    # index of the list: [input_layer, hidden[0], ..., hidden[n], output_layer]
    gen_norm: tuple[str, List[int]]  # e.g. (batch, [0, 1, 2, ..., n])
    gen_act: tuple[str, List[int]]  # e.g. (relu, [0, 2])

    rec_type: str
    rec_hidden: List[int]
    # For *_norm and *_act, the List[int] argument is a list of
    # indices which indicate an application AFTER the corresponding
    # index of the list: [input_layer, hidden[0], ..., hidden[n], output_layer]
    rec_norm: List[int]
    rec_act: tuple[str, List[int]]

    init: Dict  # jax initializer, e.g. {'init': 'normal', 'stddev': 0.1}
    opt: Dict  # optax optimizer, e.g. {'opt': 'adagrad', 'lr': 1e-1}


@dataclass
class Config:

    seed: int
    nn_lib: str

    jax: JaxConfig
    run: RunConfig
    models: ModelsConfig
