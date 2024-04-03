import hydra
from omegaconf import DictConfig, OmegaConf

from aevb.config import Config, JaxConfig, ModelsConfig, RunConfig


def create_flax_models(config: Config): ...


def create_equinox_models(config: Config): ...


def create_haiku_models(config: Config): ...


def create_aevb_engine(config: Config): ...


@hydra.main(version_base=None, config_path="./configs/", config_name="config")
def main(cfg: DictConfig):
    jax_config = JaxConfig(**cfg.jax)
    run_config = RunConfig(**cfg.run)
    models_config = ModelsConfig(**cfg.models)

    config = Config(cfg.seed, cfg.nn_lib, jax_config, run_config, models_config)
    print(config)


if __name__ == "__main__":
    main()
