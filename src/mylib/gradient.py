import torch


def gradient(
    func,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Compute the gradient of a scalar-valued function with respect to x.

    Parameters
    ----------
    func : callable
        Function mapping a single input sample of shape (N,)
        to a scalar.
    x : torch.Tensor
        Input tensor of shape (B, N).

    Returns
    -------
    torch.Tensor
        Gradient of shape (B, N).
    """
    return torch.func.vmap(torch.func.grad(func))(x)