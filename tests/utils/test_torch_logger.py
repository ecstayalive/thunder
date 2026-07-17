from __future__ import annotations

import inspect

import pytest

import thunder.utils.logging as logging_module
import torch
from thunder.core.data import attr_dataclass
from thunder.utils import logging_plugin as plugin_module
from thunder.utils.logging import (
    Image,
    Logger,
    LoggerPlugin,
    LogItem,
    LogItemRegistry,
    LogRecord,
    Scalar,
    TensorBoardLogger,
)
from thunder.utils.workspace import Workspace


@attr_dataclass(slots=True)
class SpecialItem(LogItem):
    value: int = 0


class MemoryLogger(Logger):
    registry = LogItemRegistry()

    def __init__(self, workspace: Workspace, **kwargs):
        super().__init__(workspace, **kwargs)
        self.events = []

    def init_impl(self):
        return

    @registry.register(Scalar)
    def log_scalar(self, name: str, item: Scalar, step: int):
        self.events.append((name, item.value, step))

    def close(self):
        return


class SpecialPlugin(LoggerPlugin):
    registry = LogItemRegistry()

    @registry.register(SpecialItem)
    def process_special(self, name: str, item: SpecialItem, step: int):
        return {f"{name}/value": Scalar(item.value)}


def workspace(tmp_path) -> Workspace:
    return Workspace.create(str(tmp_path), "proj", run_name="run", timestamp=False)


def test_logger_requires_explicit_log_items(tmp_path):
    logger = MemoryLogger(workspace(tmp_path))

    with pytest.raises(TypeError, match="LogItem"):
        logger.log({"loss": torch.tensor(1.0)}, step=1)


def test_logger_uses_its_own_registry_to_dispatch_items(tmp_path):
    logger = MemoryLogger(workspace(tmp_path))

    logger.log({"loss": Scalar(torch.tensor(1.0))}, step=3)

    assert logger.events == [("loss", torch.tensor(1.0), 3)]


def test_unsupported_items_skip_by_default_and_raise_in_strict_mode(tmp_path):
    logger = MemoryLogger(workspace(tmp_path))
    strict_logger = MemoryLogger(workspace(tmp_path), strict=True)

    logger.log({"frame": Image(torch.zeros(3, 8, 8))}, step=1)
    with pytest.raises(TypeError, match="does not support Image"):
        strict_logger.log({"frame": Image(torch.zeros(3, 8, 8))}, step=1)

    assert logger.events == []


def test_logger_plugin_processes_special_log_items_before_backend_dispatch(tmp_path):
    logger = MemoryLogger(workspace(tmp_path), plugins=[SpecialPlugin()])

    logger.log({"special": SpecialItem(7)}, step=5)

    assert logger.events == [("special/value", 7, 5)]


def test_logger_and_item_intervals_skip_work_before_initialization(tmp_path):
    logger = MemoryLogger(workspace(tmp_path), interval=2)

    logger.log({"loss": Scalar(1)}, step=1)
    assert logger.initialized is False
    assert logger.events == []

    logger.log({"loss": Scalar(2, interval=3)}, step=2)
    assert logger.events == []

    logger.log({"loss": Scalar(3, interval=3)}, step=6)
    assert logger.events == [("loss", 3, 6)]


def test_log_record_does_not_own_detach_lifecycle():
    value = torch.tensor(1.0, requires_grad=True)

    record = LogRecord.from_mapping({"loss": Scalar(value)})

    assert not hasattr(LogItem, "detach")
    assert not hasattr(LogRecord, "detached")
    assert record["loss"].value is value


def test_logger_module_has_no_procedural_top_level_functions():
    functions = sorted(
        name
        for name, value in vars(logging_module).items()
        if inspect.isfunction(value) and value.__module__ == logging_module.__name__
    )

    assert functions == []


def test_logger_module_has_no_log_value_toolbox():
    assert not hasattr(logging_module, "LogValue")


def test_ensure_data_documents_backend_conversion_boundary():
    doc = Logger.ensure_data.__doc__ or ""

    assert "backend" in doc.lower()
    assert "Executor.to_numpy" in doc


def test_log_item_owns_interval_validation_without_schedule_class():
    assert not hasattr(logging_module, "LogSchedule")

    with pytest.raises(ValueError, match="interval must be >= 1"):
        Scalar(1, interval=0)

    item = Scalar(1, interval=2)

    assert item.is_active(4)
    assert not item.is_active(5)
    assert not hasattr(LogItem, "on_interval")
    assert not hasattr(LogItem, "validate_interval")
    assert not hasattr(LogItem, "allows_step")
    assert not hasattr(LogItem, "should_log")
    assert not hasattr(LogItem, "due")


def test_log_record_activates_items_for_the_current_step():
    record = LogRecord.from_mapping(
        {
            "fast": Scalar(1),
            "slow": Scalar(2, interval=2),
        }
    )

    assert list(record.activate_items(1)) == ["fast"]
    assert list(record.activate_items(2)) == ["fast", "slow"]
    assert not hasattr(LogRecord, "active")


def test_cutsne_plugin_is_a_log_item_processor(tmp_path):
    from thunder.utils.logging_plugin import CuTSNEPlugin, TSNE

    plugin = CuTSNEPlugin(workspace(tmp_path), enable=False)
    item = TSNE(torch.zeros(2, 3, 4), torch.ones(2, 3))

    assert isinstance(plugin, LoggerPlugin)
    assert plugin.supports(item)
    assert "Executor" not in inspect.signature(CuTSNEPlugin.__init__).parameters


def test_utils_exports_logger_and_plugin_symbols():
    from thunder.utils import CuTSNEPlugin, Scalar, TensorBoardLogger

    assert Scalar is not None
    assert TensorBoardLogger is not None
    assert CuTSNEPlugin is not None


def test_plugin_lives_in_backend_neutral_utils_module():
    assert plugin_module.TSNE.__module__ == plugin_module.__name__
    assert plugin_module.CuTSNEPlugin.__module__ == plugin_module.__name__
    assert logging_module.Scalar is Scalar
    assert logging_module.LogItem is LogItem
    assert logging_module.Logger is Logger
    assert logging_module.TensorBoardLogger is TensorBoardLogger
