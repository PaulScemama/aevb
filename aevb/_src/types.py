from typing import Any, Iterable, Mapping

import jax
from jax.typing import ArrayLike

ArrayTree = jax.Array | Iterable["ArrayTree"] | Mapping[Any, "ArrayTree"]
ArrayLikeTree = ArrayLike | Iterable["ArrayLikeTree"] | Mapping[Any, "ArrayLikeTree"]
