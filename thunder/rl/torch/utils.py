import torch


def compute_lambda_returns(rewards, values, continues, gamma=0.99, lambda_=0.95):
    """
    Computes exponentially weighted lambda returns taking episode termination into account.

    Args:
        rewards: Tensor of shape [H, B, 1]
        values: Tensor of shape [H + 1, B, 1]
        continues: Tensor of shape [H, B, 1]. Predicted probability of NOT being done (1.0 = continue, 0.0 = done).
        gamma: float, base discount factor.
        lambda_: float, lambda parameter for weighting between n-step returns and value estimates.
    Returns:
        returns: Tensor of shape [H, B, 1]
    """
    returns = torch.zeros_like(rewards)
    last = values[-1]
    next_values = values[1:]

    for t in reversed(range(len(rewards))):
        gamma_t = gamma * continues[t]
        returns[t] = rewards[t] + gamma_t * ((1 - lambda_) * next_values[t] + lambda_ * last)
        last = returns[t]

    return returns


def compute_gae(*args, **kwargs):
    raise NotImplementedError
