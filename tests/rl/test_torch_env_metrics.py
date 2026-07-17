import torch

from thunder.rl.torch import env as env_module
from thunder.rl.torch.env import EnvMetrics
from thunder.rl.torch.operations import Play
from thunder.utils import Matrix, Scalar


def test_env_metrics_reports_zero_for_empty_log_samples_without_host_sync():
    metrics = EnvMetrics(log_on_done_only=True)
    rewards = torch.ones(2)
    dones = torch.zeros(2, dtype=torch.bool)
    info = {"log": {"upright": torch.tensor([1.0, 2.0])}}

    metrics.update(rewards, dones, dones, info)
    result = metrics.compute()

    assert "Reward/mean" in result
    assert isinstance(result["Reward/mean"], Scalar)
    assert isinstance(result["EnvLog/upright"], Scalar)
    assert torch.equal(result["EnvLog/upright"].value, torch.tensor(0.0))


def test_env_metrics_records_logs_every_step_by_default():
    metrics = EnvMetrics()
    rewards = torch.ones(2)
    dones = torch.zeros(2, dtype=torch.bool)
    info = {"log": {"upright": torch.tensor([1.0, 3.0])}}

    metrics.update(rewards, dones, dones, info)
    result = metrics.compute()

    assert isinstance(result["EnvLog/upright"], Scalar)
    assert torch.allclose(result["EnvLog/upright"].value, torch.tensor(2.0))


def test_env_metrics_compute_does_not_call_tensor_item_for_env_logs(monkeypatch):
    def fail_item(self):
        raise AssertionError("Tensor.item synchronizes EnvLog metrics")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    metrics = EnvMetrics()
    rewards = torch.ones(2)
    dones = torch.zeros(2, dtype=torch.bool)
    info = {"log": {"upright": torch.tensor([1.0, 3.0])}}

    metrics.update(rewards, dones, dones, info)
    result = metrics.compute()

    assert isinstance(result["EnvLog/upright"], Scalar)
    assert torch.allclose(result["EnvLog/upright"].value, torch.tensor(2.0))


def test_env_metrics_ignores_nonfinite_masked_log_values():
    metrics = EnvMetrics(log_on_done_only=True)
    rewards = torch.ones(3)
    terminated = torch.tensor([True, False, True])
    timeouts = torch.zeros(3, dtype=torch.bool)
    info = {"log": {"upright": torch.tensor([1.0, torch.nan, 3.0])}}

    metrics.update(rewards, terminated, timeouts, info)
    result = metrics.compute()

    assert isinstance(result["EnvLog/upright"], Scalar)
    assert torch.allclose(result["EnvLog/upright"].value, torch.tensor(2.0))


def test_env_metrics_reports_transition_steps_per_second(monkeypatch):
    times = iter([10.0, 12.0])
    monkeypatch.setattr(env_module.time, "perf_counter", lambda: next(times))
    metrics = EnvMetrics()
    dones = torch.zeros(3, dtype=torch.bool)

    metrics.update(torch.ones(3), dones, dones, None)
    result = metrics.compute()

    assert isinstance(result["Step/fps"], Scalar)
    assert result["Step/fps"].value == 1.5


def test_play_observation_metrics_preserve_vector_semantics():
    play = Play.__new__(Play)
    value = torch.tensor([1.0, 2.0, 3.0])
    play.observation_terms = lambda: {"policy/joint_pos": value}

    result = Play.observation_metrics(play)

    item = result["Observation/policy/joint_pos"]
    assert isinstance(item, Matrix)
    assert torch.equal(item.values, value)


def test_play_observation_metrics_keep_per_env_matrix():
    play = Play.__new__(Play)
    value = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2 envs, 3 features)
    play.observation_terms = lambda: {"policy/joint_pos": value}

    result = Play.observation_metrics(play)

    item = result["Observation/policy/joint_pos"]
    assert isinstance(item, Matrix)
    assert torch.equal(item.values, value)  # not reduced over the env dimension


def test_play_reward_terms_index_terms_across_envs():
    play = Play.__new__(Play)

    class FakeRewardManager:
        _step_reward = torch.tensor(
            [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]
        )  # (3 envs, 2 terms)
        active_terms = ["alpha", "beta"]

    play.reward_manager = FakeRewardManager()

    terms = Play.reward_terms(play)

    assert torch.equal(terms["alpha"], torch.tensor([10.0, 11.0, 12.0]))
    assert torch.equal(terms["beta"], torch.tensor([20.0, 21.0, 22.0]))
