import torch

from mylib import MLP


def test_mlp_output_shape():
    model = MLP(
        in_dim=2,
        out_dim=1,
        hid_dim=32,
        n_layers=3,
    )

    x = torch.randn(10, 2)
    y = model(x)

    assert y.shape == (10, 1)


def test_mlp_gradient():
    model = MLP(in_dim=2, out_dim=1)

    x = torch.randn(5, 2, requires_grad=True)
    y = model(x)

    y.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
