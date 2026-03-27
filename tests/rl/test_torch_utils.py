import importlib.util
from pathlib import Path

import torch

_UTILS_PATH = Path(__file__).resolve().parents[2] / "thunder" / "rl" / "torch" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("thunder_rl_torch_utils_test", _UTILS_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
compute_lambda_returns = _MODULE.compute_lambda_returns


def test_compute_lambda_returns_bootstraps_with_next_value():
    rewards = torch.tensor([[[1.0]], [[2.0]]])
    values = torch.tensor([[[10.0]], [[20.0]], [[30.0]]])
    continues = torch.ones_like(rewards)

    returns = compute_lambda_returns(rewards, values, continues, gamma=1.0, lambda_=0.0)

    expected = torch.tensor([[[21.0]], [[32.0]]])
    assert torch.allclose(returns, expected)
