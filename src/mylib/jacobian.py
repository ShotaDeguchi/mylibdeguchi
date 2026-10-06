import torch


def jacobian(
    func,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Jacobian of func with respect to x.

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
        Jacobian of shape (B, M, N).
    """
    return torch.func.vmap(torch.func.jacrev(func))(x)


