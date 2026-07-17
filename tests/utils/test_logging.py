from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.utils._cxx_pytree as pytree

from thunder.core import Batch, ExecutionContext, Executor, Objective
from thunder.core.data import attr_dataclass, AttrData
from thunder.utils import logging as logging_module
from thunder.utils.logging import (
    AsyncLogger,
    Histogram,
    Logger,
    LogItem,
    LogRecord,
    Matrix,
    PlotJugglerLogger,
    Scalar,
    TensorBoardLogger,
)


def test_log_items_live_in_backend_neutral_logging_module():
    item = Scalar(torch.tensor(1.0))

    assert isinstance(item, LogItem)
    assert isinstance(item, AttrData)
    assert Scalar.__module__ == logging_module.__name__
    assert item.value.equal(torch.tensor(1.0))
    assert item.interval == 1


def test_log_items_are_pytree_data():
    item = Histogram(torch.ones(2), interval=3)

    mapped = pytree.tree_map(
        lambda leaf: leaf + 1 if isinstance(leaf, torch.Tensor) else leaf,
        item,
    )

    assert isinstance(mapped, Histogram)
    assert torch.equal(mapped.values, torch.full((2,), 2.0))
    assert mapped.interval == 3


def test_vector_log_item_represents_feature_values():
    item = Matrix(torch.arange(3.0), labels=("x", "y", "z"))

    assert isinstance(item, LogItem)
    assert item.values.equal(torch.tensor([0.0, 1.0, 2.0]))
    assert item.labels == ("x", "y", "z")


def _matrix_logger():
    class _MatrixLogger(Logger):
        def init_impl(self):
            return None

        def close(self):
            return None

    return _MatrixLogger(None)


def test_matrix_items_expand_vector_and_matrix_per_element():
    logger = _matrix_logger()

    vector = dict(logger.matrix_items("Reward", Matrix(torch.tensor([1.0, 2.0, 3.0]))))
    assert vector == {"Reward/0": 1.0, "Reward/1": 2.0, "Reward/2": 3.0}

    grid = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (2 envs, 2 features)
    expanded = dict(logger.matrix_items("Obs", Matrix(grid)))
    assert expanded == {
        "Obs/0/0": 1.0,
        "Obs/0/1": 2.0,
        "Obs/1/0": 3.0,
        "Obs/1/1": 4.0,
    }


def test_matrix_items_label_names_the_last_axis():
    logger = _matrix_logger()
    grid = torch.tensor([[10.0, 20.0], [11.0, 21.0]])  # (2 envs, 2 terms)

    labeled = dict(
        logger.matrix_items("RewardTerm", Matrix(grid, labels=("alpha", "beta")))
    )
    assert labeled == {
        "RewardTerm/0/alpha": 10.0,
        "RewardTerm/0/beta": 20.0,
        "RewardTerm/1/alpha": 11.0,
        "RewardTerm/1/beta": 21.0,
    }


def test_matrix_items_rejects_labels_mismatched_with_last_axis():
    logger = _matrix_logger()
    try:
        list(
            logger.matrix_items(
                "RewardTerm", Matrix(torch.zeros(2, 3), labels=("a", "b"))
            )
        )
    except ValueError as exc:
        assert "Matrix labels" in str(exc)
        assert "last-axis" in str(exc)
    else:
        raise AssertionError("matrix_items accepted labels mismatched with last axis")


class _FakeUdpSocket:
    """Mimics a UDP socket, enforcing the IPv4 datagram payload ceiling."""

    MAX = 65507

    def __init__(self):
        self.sent = []

    def sendto(self, blob, addr):
        if len(blob) > self.MAX:
            raise OSError(90, "Message too long")
        self.sent.append(blob)

    def close(self):
        pass


def test_plotjuggler_sends_small_record_as_single_datagram():
    logger = PlotJugglerLogger(enable=True)
    logger.sock = _FakeUdpSocket()
    logger._pending = {"Reward/0": 0.5, "Reward/1": 0.6}

    logger.end_record(step=3)

    assert len(logger.sock.sent) == 1
    payload = json.loads(logger.sock.sent[0])
    assert payload == {"Reward/0": 0.5, "Reward/1": 0.6, "step": 3}


def test_plotjuggler_chunks_oversized_record_instead_of_dropping_it():
    logger = PlotJugglerLogger(enable=True)
    logger.sock = _FakeUdpSocket()
    # Per-env expansion at non-trivial num_envs blows past one datagram.
    logger._pending = {f"Obs/{i // 50}/{i % 50}": float(i) for i in range(8000)}

    logger.end_record(step=7)

    sock = logger.sock
    assert len(sock.sent) > 1  # split across datagrams, not silently dropped
    assert all(len(b) <= _FakeUdpSocket.MAX for b in sock.sent)
    delivered = {}
    for blob in sock.sent:
        obj = json.loads(blob)
        assert obj["step"] == 7
        delivered.update({k: v for k, v in obj.items() if k != "step"})
    assert len(delivered) == 8000  # every field reaches PlotJuggler


def test_log_item_validates_interval_after_attr_data_init():
    try:
        Scalar(torch.tensor(1.0), interval=0)
    except ValueError as exc:
        assert "interval must be >= 1" in str(exc)
    else:
        raise AssertionError("Scalar accepted an invalid interval")


def test_log_record_accepts_backend_neutral_log_items():
    record = LogRecord.from_mapping(
        {
            "loss": Scalar(torch.tensor(1.0)),
            "weights": Histogram(torch.ones(3), interval=2),
        }
    )

    assert list(record.activate_items(1)) == ["loss"]
    assert list(record.activate_items(2)) == ["loss", "weights"]


def test_attr_dataclass_preserves_keyword_only_fields_for_positional_init():
    @attr_dataclass(slots=True, kw_only=True)
    class Payload(AttrData):
        interval: int = 1

        def __post_init__(self):
            if self.interval < 1:
                raise ValueError("bad interval")

    @attr_dataclass(slots=True)
    class Child(Payload):
        value: int = 0

    item = Child(7, interval=3)

    assert item.value == 7
    assert item.interval == 3


def test_utils_torch_namespace_has_been_removed():
    thunder_root = Path(__file__).parents[2] / "thunder"

    assert not (thunder_root / "utils" / "torch").exists()


def test_concrete_loggers_live_in_backend_neutral_logging_module():
    assert TensorBoardLogger.__module__ == logging_module.__name__
    assert PlotJugglerLogger.__module__ == logging_module.__name__
    assert AsyncLogger.__module__ == logging_module.__name__


def test_logger_uses_module_executor_class_directly(tmp_path):
    class MemoryLogger(Logger):
        def init_impl(self):
            return

        def close(self):
            return

    logger = MemoryLogger(
        logging_module.Workspace.create(
            str(tmp_path), "proj", run_name="run", timestamp=False
        )
    )

    assert "Executor" not in inspect.signature(Logger.__init__).parameters
    assert (
        "Executor"
        not in inspect.signature(logging_module.LoggerPlugin.__init__).parameters
    )
    assert "Executor" not in inspect.signature(TensorBoardLogger.__init__).parameters
    assert logging_module.Executor is Executor
    assert logger.as_scalar(torch.tensor(4.0)) == 4.0


def test_executor_detach_removes_torch_autograd_graph_without_moving_device():
    leaf = torch.tensor(2.0, requires_grad=True)
    value = leaf * 3.0

    detached = Executor.detach({"value": value})

    assert detached["value"].requires_grad is False
    assert detached["value"].grad_fn is None
    assert detached["value"].device == value.device
    assert torch.equal(detached["value"], torch.tensor(6.0, device=value.device))


def test_async_logger_is_a_logger():
    logger = AsyncLogger([], interval=2)

    assert isinstance(logger, Logger)
    assert logger.workspace is None
    assert not logger.is_active(1)
    assert logger.is_active(2)
    logger.close()


def test_async_logger_strict_rejects_tensors_with_autograd_graph():
    class FakeProcess:
        def is_alive(self):
            return True

    class FakeQueue:
        def __init__(self):
            self.packet = None

        def put_nowait(self, packet):
            self.packet = packet

    logger = AsyncLogger([], strict=True)
    logger._process = FakeProcess()
    logger._queue = FakeQueue()
    base = torch.tensor(1.0, requires_grad=True)
    value = base * 2.0

    try:
        logger.log({"loss": Scalar(value), "features": Matrix(torch.stack([value]))}, 1)
    except ValueError as exc:
        assert "loss.value" in str(exc)
        assert "detach" in str(exc)
    else:
        raise AssertionError("AsyncLogger accepted a LogItem with autograd graph")

    assert logger._queue.packet is None
    logger.close()


def test_objective_evaluate_returns_explicit_scalar_log_items():
    class TinyObjective(Objective):
        def compute(self, ctx):
            leaf = torch.tensor(2.0, requires_grad=True)
            value = leaf * 2.0
            return value, {"mean": Scalar(value.detach())}

    objective = TinyObjective(name="tiny")
    _, metrics = objective.evaluate(ExecutionContext.create(None, None, None, {}))

    assert isinstance(metrics["tiny/loss"], Scalar)
    assert isinstance(metrics["tiny/weighted_loss"], Scalar)
    assert isinstance(metrics["tiny/mean"], Scalar)
    assert torch.equal(metrics["tiny/loss"].value, torch.tensor(4.0))
    assert metrics["tiny/loss"].value.requires_grad is False
    assert metrics["tiny/loss"].value.grad_fn is None
    assert metrics["tiny/weighted_loss"].value.requires_grad is False
    assert metrics["tiny/weighted_loss"].value.grad_fn is None


def test_log_item_construction_does_not_break_torch_compile_graph():
    dynamo = pytest.importorskip("torch._dynamo")

    def build_metrics(x):
        loss = x.square().mean()
        return {"loss": Scalar(loss.detach())}

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to reproduce the LogItem Dynamo graph break.")

    explanation = dynamo.explain(build_metrics)(torch.ones(8, device="cuda"))

    assert explanation.break_reasons == []


def test_dynamic_attr_data_access_does_not_break_torch_compile_graph():
    dynamo = pytest.importorskip("torch._dynamo")
    root = Batch(cache=Batch(initial=Batch(policy_carry=torch.ones(2))))

    def read_carry(root):
        return root.cache.initial.policy_carry + 1

    dynamo.reset()
    explanation = dynamo.explain(read_carry)(root)
    assert explanation.graph_break_count == 0
    assert explanation.break_reasons == []

    out = torch.compile(read_carry, backend="eager", fullgraph=True)(root)
    assert torch.equal(out, torch.full((2,), 2.0))
