import torch

from mylib import jacobian


def test_jacobian():
    def model(x):
        return torch.stack([
            x[0]**2 + x[1],
            x[0] * x[1],
        ])

    x = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    J = jacobian(model, x)

    expected = torch.tensor([
        [
            [2.0, 1.0],
            [2.0, 1.0],
        ],
        [
            [6.0, 1.0],
            [4.0, 3.0],
        ],
    ])

    assert J.shape == (2, 2, 2)
    assert torch.allclose(J, expected)


def test_jacobian_second_derivative():
    def model(x):
        return torch.stack([
            x[0]**3 + x[1],
        ])

    x = torch.tensor(
        [[1.0, 2.0],
         [2.0, 3.0]],
        requires_grad=True,
    )

    J = jacobian(model, x)

    # J: (N, 1, 2)
    # d/dx1 (x1^3 + x2) = 3 x1^2
    expected = torch.tensor([
        [[3.0, 1.0]],
        [[12.0, 1.0]],
    ])

    assert torch.allclose(J, expected)

    # Second derivative with respect to x1
    d2 = torch.autograd.grad(
        J[:, 0, 0].sum(),
        x,
        create_graph=True,
    )[0]

    expected_d2 = torch.tensor([
        [6.0, 0.0],
        [12.0, 0.0],
    ])

    assert torch.allclose(d2, expected_d2)