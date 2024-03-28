try:
    import equinox
except ModuleNotFoundError:
    message = "Please install equinox to use equinox networks."
    raise ModuleNotFoundError(message)

from aevb._src.eqx_util import (EqxMLPDecoder, EqxMLPEncoder, batch_model,
                                init_apply_eqx_model)
