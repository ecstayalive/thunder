import torch

__all__ = ["gaussian_kl_divergence"]


@torch.jit.script
def gaussian_kl_divergence(mu1, sigma1, mu2, sigma2):
    kl = torch.sum(
        torch.log(sigma2 / sigma1 + 1.0e-5)
        + (torch.square(sigma1) + torch.square(mu1 - mu2)) / (2.0 * torch.square(sigma2))
        - 0.5,
        dim=-1,
    )
    return kl.mean()
