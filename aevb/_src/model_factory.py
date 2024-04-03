import aevb._src.util as util


class ModelFactory(util.ObjectFactory): ...


factory = ModelFactory()
factory.register_builder("flax-mlp-encoder")
