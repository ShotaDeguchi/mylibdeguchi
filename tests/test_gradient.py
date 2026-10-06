import torch

from mylib import gradient


def test_gradient():
    def func(x):
        return x[0]**2 + x[1]**3

    x = torch.tensor([
        [1.0, 2.0],
        [2.0, 3.0],
    ])

    grad = gradient(func, x)

    expected = torch.tensor([
        [2.0, 12.0],
        [4.0, 27.0],
    ])

    assert grad.shape == (2, 2)
    assert torch.allclose(grad, expected)