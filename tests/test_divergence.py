import torch

from mylib import divergence


def test_divergence():
    def func(x):
        return torch.stack([
            x[0]**2 + x[1],
            x[0] * x[1] + x[1]**3,
        ])

    x = torch.tensor([
        [1.0, 2.0],
        [2.0, 3.0],
    ])

    div = divergence(func, x)

    expected = torch.tensor([
        15.0,  # 3*1 + 3*2^2
        33.0,  # 3*2 + 3*3^2
    ])

    assert div.shape == (2,)
    assert torch.allclose(div, expected)