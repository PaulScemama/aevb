try:
    import flax
except ModuleNotFoundError:
    message = "Please install flax to use flax networks."
    raise ModuleNotFoundError(message)

from aevb._src.flax_util import (FlaxMLPDecoder, FlaxMLPEncoder,
                                 init_apply_flax_model)
