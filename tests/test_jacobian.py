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

    jac = jacobian(model, x)

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

    assert jac.shape == (2, 2, 2)
    assert torch.allclose(jac, expected)


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

    jac = jacobian(model, x)

    # jac: (N, 1, 2)
    # d/dx1 (x1^3 + x2) = 3 x1^2
    expected = torch.tensor([
        [[3.0, 1.0]],
        [[12.0, 1.0]],
    ])

    assert torch.allclose(jac, expected)

    # Second derivative with respect to x1
    j2 = torch.autograd.grad(
        jac[:, 0, 0].sum(),
        x,
        create_graph=True,
    )[0]

    expected_j2 = torch.tensor([
        [6.0, 0.0],
        [12.0, 0.0],
    ])

    assert torch.allclose(j2, expected_j2)