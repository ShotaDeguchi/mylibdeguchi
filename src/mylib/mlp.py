import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 1,
        hid_dim: int = 64,
        n_layers: int = 2,
        act_fn: nn.Module = nn.Tanh
    ):
        """
        MLP.

        With `n_layers` = L, the network has L-1 hidden layers
        followed by a linear output layer:
            z^(0) = x
            z^(l) = act_fn(W^(l) z^(l-1) + b^(l)),  l = 1, ..., L-1
            z^(L) = W^(L) z^(L-1) + b^(L)

        Parameters
        ----------
        in_dim : int
            Input dimension.
        out_dim : int
            Output dimension.
        hid_dim : int
            Hidden dimension.
        n_layers : int
            Number of layers, L.
            The input is labeled as layer 0, the output is labeled as layer L.
        act_fn : nn.Module
            Activation function for hidden layers.
        """
        super().__init__()
        assert n_layers >= 1, "n_layers must be >= 1"

        dims = [in_dim] + [hid_dim] * (n_layers - 1) + [out_dim]
        layers = []
        for l in range(1, n_layers+1):
            layers.append(nn.Linear(dims[l-1], dims[l], bias=True))
            if l < n_layers:  # output layer is linear
                layers.append(act_fn())
        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.layers(x)
