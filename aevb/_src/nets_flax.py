try:
    import flax.linen as nn 
except ModuleNotFoundError:
    message = "Please install flax to use flax networks."


from typing import List, Callable

import flax.linen as nn


class FlaxMLP(nn.Module):
    hidden: List[int]
    activation: Callable

    @nn.compact
    def __call__(self, x):
        num_layers = len(self.hidden)
        for i, h in enumerate(self.hidden):
            x = nn.Dense(h)(x)
            if i < num_layers:
                x = self.activation(x)
        return x
    
