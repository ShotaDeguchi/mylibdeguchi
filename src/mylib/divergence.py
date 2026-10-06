import torch

from .jacobian import jacobian


def divergence(
    func,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Compute the divergence of func with respect to x.

    Parameters
    ----------
    func : callable
        Function mapping a single input sample of shape (N,)
        to an output of shape (M,).
    x : torch.Tensor
        Input tensor of shape (B, N).

    Returns
    -------
    torch.Tensor
        Divergence of shape (B,).
    """
    J = jacobian(func, x)
    return J.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
