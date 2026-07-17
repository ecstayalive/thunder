from __future__ import annotations

import atexit
import json
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, final, Mapping

import numpy as np

from thunder.core.data import attr_dataclass, AttrData
from thunder.core.executor import Executor

from .workspace import Workspace


@attr_dataclass(slots=True, kw_only=True)
class LogItem(AttrData):
    """Backend-neutral typed log payload."""

    interval: int = 1

    def __post_init__(self):
        if self.interval < 1:
            raise ValueError(f"interval must be >= 1, got {self.interval!r}")

    def is_active(self, step: int) -> bool:
        return step % self.interval == 0


@attr_dataclass(slots=True)
class Scalar(LogItem):
    value: Any = None


@attr_dataclass(slots=True)
class Image(LogItem):
    data: Any = None
    dataformats: str = "CHW"


@attr_dataclass(slots=True)
class Histogram(LogItem):
    values: Any = None
    bins: str | int = "tensorflow"


@attr_dataclass(slots=True)
class Matrix(LogItem):
    values: Any = None
    labels: tuple[str, ...] | None = None


@attr_dataclass(slots=True)
class Figure(LogItem):
    figure: Any = None
    close: bool = True


@attr_dataclass(slots=True)
class Text(LogItem):
    text: str = ""


@attr_dataclass(slots=True)
class Artifact(LogItem):
    path: str | Path = ""
    name: str | None = None
    type: str = "artifact"
    metadata: Dict[str, Any] | None = None


class LogRecord(dict[str, LogItem]):
    @classmethod
    def from_mapping(cls, metrics: Mapping[str, LogItem]):
        if isinstance(metrics, cls):
            return cls(metrics)
        if not isinstance(metrics, Mapping):
            raise TypeError(
                f"Logger expects a mapping of str -> LogItem, got {type(metrics).__name__}."
            )
        validated = cls()
        for name, item in metrics.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"Log item names must be str, got {type(name).__name__}."
                )
            if not isinstance(item, LogItem):
                raise TypeError(
                    f"Logger expects every metric to be an explicit LogItem. "
                    f"Metric {name!r} got {type(item).__name__}."
                )
            validated[name] = item
        return validated

    def activate_items(self, step: int):
        return type(self)(
            {name: item for name, item in self.items() if item.is_active(step)}
        )


class LogItemRegistry:
    """Per-logger registry mapping LogItem types to handlers."""

    def __init__(self):
        self._handlers: dict[type[LogItem], Callable] = {}

    def register(self, item_type: type[LogItem]):
        if not isinstance(item_type, type) or not issubclass(item_type, LogItem):
            raise TypeError("LogItemRegistry.register expects a LogItem subclass.")

        def decorator(handler: Callable):
            self._handlers[item_type] = handler
            return handler

        return decorator

    def get(self, item: LogItem | type[LogItem]):
        item_type = item if isinstance(item, type) else type(item)
        for cls in item_type.__mro__:
            handler = self._handlers.get(cls)
            if handler is not None:
                return handler
        return None


class LoggerPlugin(ABC):
    """Transforms special LogItems into ordinary LogItems before backend logging."""

    registry = LogItemRegistry()

    def __init__(
        self,
        enable: bool = True,
        interval: int = 1,
        strict: bool = False,
    ):
        self.enable = enable
        if interval < 1:
            raise ValueError(f"interval must be >= 1, got {interval!r}")
        self.interval = interval
        self.strict = strict

    def supports(self, item: LogItem) -> bool:
        return self.registry.get(item) is not None

    def is_active(self, step: int) -> bool:
        return step % self.interval == 0

    @final
    def process(self, name: str, item: LogItem, step: int) -> LogRecord:
        if not self.enable or not self.is_active(step):
            return LogRecord()
        if not item.is_active(step):
            return LogRecord()
        handler = self.registry.get(item)
        if handler is None:
            if self.strict:
                raise TypeError(
                    f"{type(self).__name__} does not support {type(item).__name__}: {name}"
                )
            return LogRecord({name: item})
        result = handler(self, name, item, step)
        if result is None:
            return LogRecord()
        return LogRecord.from_mapping(result)


class Logger(ABC):
    registry = LogItemRegistry()

    def __init__(
        self,
        workspace: Workspace | None,
        enable: bool = True,
        interval: int = 1,
        strict: bool = False,
        plugins: list[LoggerPlugin] | None = None,
    ):
        self.workspace = workspace
        self.enable = enable
        if interval < 1:
            raise ValueError(f"interval must be >= 1, got {interval!r}")
        self.interval = interval
        self.strict = strict
        self.plugins = list(plugins or [])
        self.initialized = False
        self._logger = None

    def is_active(self, step: int) -> bool:
        return step % self.interval == 0

    def log(self, metrics: Mapping[str, LogItem], step: int):
        if not self.enable or not self.is_active(step):
            return
        record = self._process_plugins(metrics, step)
        if not record:
            return
        if not self.initialized:
            self.init()
        self.begin_record(step)
        try:
            for name, item in record.items():
                if item.is_active(step):
                    self.emit(name, item, step)
        finally:
            self.end_record(step)

    @final
    def init(self):
        if self.workspace is not None:
            self.workspace.ensure()
        self.init_impl()
        self.initialized = True

    def _process_plugins(self, metrics: Mapping[str, LogItem], step: int) -> LogRecord:
        record = LogRecord()
        for name, item in LogRecord.from_mapping(metrics).activate_items(step).items():
            plugin = self._plugin_for(item)
            if plugin is None:
                record[name] = item
            else:
                record.update(plugin.process(name, item, step))
        return record

    def _plugin_for(self, item: LogItem):
        for plugin in self.plugins:
            if plugin.supports(item):
                return plugin
        return None

    def emit(self, name: str, item: LogItem, step: int):
        handler = self.registry.get(item)
        if handler is None:
            return self.unsupported(name, item)
        return handler(self, name, item, step)

    def unsupported(self, name: str, item: LogItem):
        if self.strict:
            raise TypeError(
                f"{type(self).__name__} does not support {type(item).__name__}: {name}"
            )
        return None

    def as_scalar(self, value):
        if isinstance(value, (bool, int, float, str)):
            return value
        value = self.ensure_data(value)
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(
                    f"Scalar expects a single value array, got shape {value.shape}."
                )
            return value.item()
        if isinstance(value, np.generic):
            return value.item()
        try:
            return float(value)
        except Exception as exc:
            raise TypeError(
                f"Cannot convert {type(value).__name__} to scalar."
            ) from exc

    def ensure_data(self, value):
        """Prepare backend-native arrays for logger writers.

        Non-primitive values are converted with Executor.to_numpy at the logging
        boundary. Depending on the backend, this may move device data to host
        memory and synchronize outstanding work.
        """
        if value is None or isinstance(value, (bool, int, float, str, bytes, Path)):
            return value
        if isinstance(value, (np.ndarray, np.generic)):
            return value
        try:
            return Executor.to_numpy(value)
        except Exception:
            return value

    def matrix_items(self, name: str, item: Matrix):
        """Expand a matrix into one scalar per element.

        Every element of ``item.values`` is yielded as ``(key, scalar)`` where
        ``key`` encodes the element's full multi-dimensional index, e.g. a
        ``(num_envs, feature)`` tensor yields ``name/{env}/{feature}``. A 1-D
        vector is the degenerate case and yields ``name/{index}``. ``labels``,
        when given, names the last axis instead of using numeric indices.
        """
        values = np.asarray(self.ensure_data(item.values))
        labels = item.labels
        if labels is not None:
            last = values.shape[-1] if values.ndim else 0
            if len(labels) != last:
                raise ValueError(
                    f"Matrix labels for {name!r} must match the last-axis size "
                    f"{last}, got {len(labels)}."
                )
        if values.ndim == 0:
            yield name, self.as_scalar(values)
            return
        for index in np.ndindex(values.shape):
            *lead, last = index
            tail = labels[last] if labels is not None else str(last)
            key = "/".join((name, *(str(i) for i in lead), tail))
            yield key, self.as_scalar(values[index])

    def begin_record(self, step: int):
        return None

    def end_record(self, step: int):
        return None

    @abstractmethod
    def init_impl(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class TensorBoardLogger(Logger):
    registry = LogItemRegistry()

    def init_impl(self):
        from torch.utils.tensorboard import SummaryWriter

        self._logger = SummaryWriter(log_dir=str(self.workspace.run_dir))

    @registry.register(Scalar)
    def log_scalar(self, name: str, item: Scalar, step: int):
        self._logger.add_scalar(name, self.as_scalar(item.value), global_step=step)

    @registry.register(Image)
    def log_image(self, name: str, item: Image, step: int):
        self._logger.add_image(
            name,
            self.ensure_data(item.data),
            global_step=step,
            dataformats=item.dataformats,
        )

    @registry.register(Histogram)
    def log_histogram(self, name: str, item: Histogram, step: int):
        self._logger.add_histogram(
            name, self.ensure_data(item.values), global_step=step, bins=item.bins
        )

    @registry.register(Matrix)
    def log_matrix(self, name: str, item: Matrix, step: int):
        for key, value in self.matrix_items(name, item):
            self._logger.add_scalar(key, value, global_step=step)

    @registry.register(Figure)
    def log_figure(self, name: str, item: Figure, step: int):
        self._logger.add_figure(name, item.figure, global_step=step, close=item.close)

    @registry.register(Text)
    def log_text(self, name: str, item: Text, step: int):
        self._logger.add_text(name, item.text, global_step=step)

    def close(self):
        if self._logger is not None:
            self._logger.close()


class WandbLogger(Logger):
    registry = LogItemRegistry()

    def __init__(
        self,
        workspace: Workspace,
        config: Dict | None = None,
        enable: bool = True,
        interval: int = 1,
        strict: bool = False,
        plugins: list[LoggerPlugin] | None = None,
        **kwargs,
    ):
        super().__init__(
            workspace,
            enable=enable,
            interval=interval,
            strict=strict,
            plugins=plugins,
        )
        self.config = config
        self.kwargs = kwargs
        self._pending = None
        self._wandb = None

    def init_impl(self):
        import wandb

        self._wandb = wandb
        self._logger = wandb.init(
            project=self.workspace.project,
            name=self.workspace.run_name,
            dir=str(self.workspace.run_dir),
            config=self.config,
            resume="allow",
            reinit=True,
            **self.kwargs,
        )

    def begin_record(self, step: int):
        self._pending = {}

    def end_record(self, step: int):
        if self._pending:
            self._logger.log(self._pending, step=step)
        self._pending = None

    @registry.register(Scalar)
    def log_scalar(self, name: str, item: Scalar, step: int):
        self._pending[name] = self.as_scalar(item.value)

    @registry.register(Image)
    def log_image(self, name: str, item: Image, step: int):
        self._pending[name] = self._wandb.Image(self.ensure_data(item.data))

    @registry.register(Histogram)
    def log_histogram(self, name: str, item: Histogram, step: int):
        self._pending[name] = self._wandb.Histogram(self.ensure_data(item.values))

    @registry.register(Matrix)
    def log_matrix(self, name: str, item: Matrix, step: int):
        self._pending.update(dict(self.matrix_items(name, item)))

    @registry.register(Figure)
    def log_figure(self, name: str, item: Figure, step: int):
        self._pending[name] = self._wandb.Image(item.figure)

    @registry.register(Text)
    def log_text(self, name: str, item: Text, step: int):
        self._pending[name] = item.text

    @registry.register(Artifact)
    def log_artifact(self, name: str, item: Artifact, step: int):
        artifact_name = item.name or Path(item.path).stem
        artifact = self._wandb.Artifact(
            artifact_name, type=item.type, metadata=item.metadata
        )
        artifact.add_file(str(item.path))
        self._logger.log_artifact(artifact)

    def close(self):
        if self._logger is not None:
            self._logger.finish()


class SwanLabLogger(Logger):
    registry = LogItemRegistry()

    def __init__(
        self,
        workspace: Workspace,
        config: Dict | None = None,
        enable: bool = True,
        interval: int = 1,
        strict: bool = False,
        plugins: list[LoggerPlugin] | None = None,
        **kwargs,
    ):
        super().__init__(
            workspace,
            enable=enable,
            interval=interval,
            strict=strict,
            plugins=plugins,
        )
        self.config = config
        self.kwargs = kwargs
        self._pending = None
        self._swanlab = None

    def init_impl(self):
        import swanlab

        self._swanlab = swanlab
        init_kwargs = {
            "project": self.workspace.project,
            "name": self.workspace.run_name,
            "work_dir": str(self.workspace.run_dir),
            "config": self.config,
        }
        init_kwargs.update(self.kwargs)
        try:
            self._logger = swanlab.init(**init_kwargs)
        except TypeError:
            init_kwargs["experiment_name"] = init_kwargs.pop("name")
            self._logger = swanlab.init(**init_kwargs)

    def begin_record(self, step: int):
        self._pending = {}

    def end_record(self, step: int):
        if self._pending:
            if hasattr(self._logger, "log"):
                self._logger.log(self._pending, step=step)
            else:
                self._swanlab.log(self._pending, step=step)
        self._pending = None

    @registry.register(Scalar)
    def log_scalar(self, name: str, item: Scalar, step: int):
        self._pending[name] = self.as_scalar(item.value)

    @registry.register(Image)
    def log_image(self, name: str, item: Image, step: int):
        image_cls = getattr(self._swanlab, "Image", None)
        data = self.ensure_data(item.data)
        self._pending[name] = image_cls(data) if image_cls else data

    @registry.register(Histogram)
    def log_histogram(self, name: str, item: Histogram, step: int):
        hist_cls = getattr(self._swanlab, "Histogram", None)
        values = self.ensure_data(item.values)
        self._pending[name] = hist_cls(values) if hist_cls else values

    @registry.register(Matrix)
    def log_matrix(self, name: str, item: Matrix, step: int):
        self._pending.update(dict(self.matrix_items(name, item)))

    @registry.register(Figure)
    def log_figure(self, name: str, item: Figure, step: int):
        image_cls = getattr(self._swanlab, "Image", None)
        self._pending[name] = image_cls(item.figure) if image_cls else item.figure

    @registry.register(Text)
    def log_text(self, name: str, item: Text, step: int):
        self._pending[name] = item.text

    def close(self):
        if self._logger is not None and hasattr(self._logger, "finish"):
            self._logger.finish()
        elif self._swanlab is not None and hasattr(self._swanlab, "finish"):
            self._swanlab.finish()


class PlotJugglerLogger(Logger):
    registry = LogItemRegistry()

    # Stay safely below the ~65507-byte IPv4 UDP datagram payload ceiling.
    MAX_DATAGRAM_BYTES = 60000

    def __init__(
        self,
        workspace: Workspace | None = None,
        host: str = "127.0.0.1",
        port: int = 9870,
        enable: bool = True,
        interval: int = 1,
        strict: bool = False,
        plugins: list[LoggerPlugin] | None = None,
    ):
        if workspace is None:
            workspace = Workspace(root="/tmp/Thunder/logs", project="plotjuggler")
        super().__init__(
            workspace,
            enable=enable,
            interval=interval,
            strict=strict,
            plugins=plugins,
        )
        self.host = host
        self.port = port
        self.sock = None
        self._pending = None
        self._warned_send = False
        self._warned_chunk = False

    def init_impl(self):
        import socket

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(
            f"[PlotJugglerLogger] UDP logger initialized. Streaming JSON to {self.host}:{self.port}"
        )

    def begin_record(self, step: int):
        self._pending = {}

    def end_record(self, step: int):
        if self.sock is None or not self._pending:
            self._pending = None
            return
        payload = dict(self._pending)
        self._pending = None
        payload["step"] = step
        blob = json.dumps(payload).encode("utf-8")
        if len(blob) <= self.MAX_DATAGRAM_BYTES:
            self._sendto(blob)
        else:
            self._send_chunked(payload, step)

    def _sendto(self, blob: bytes):
        try:
            self.sock.sendto(blob, (self.host, self.port))
        except OSError as exc:
            if not self._warned_send:
                self._warned_send = True
                print(
                    f"\033[93m[PlotJugglerLogger] UDP send failed ({len(blob)} bytes): "
                    f"{exc}. Reduce the number of streamed fields "
                    "(e.g. lower env.num_envs during play).\033[0m"
                )

    def _send_chunked(self, payload: dict, step: int):
        if not self._warned_chunk:
            self._warned_chunk = True
            print(
                f"\033[93m[PlotJugglerLogger] record has {len(payload) - 1} fields and "
                "exceeds one UDP datagram; splitting across packets. Lower "
                "env.num_envs for smoother live play.\033[0m"
            )
        chunk: dict = {}
        size = 2  # account for the enclosing braces
        for key, value in payload.items():
            if key == "step":
                continue
            entry = len(json.dumps({key: value})) + 1
            if chunk and size + entry > self.MAX_DATAGRAM_BYTES:
                chunk["step"] = step
                self._sendto(json.dumps(chunk).encode("utf-8"))
                chunk = {}
                size = 2
            chunk[key] = value
            size += entry
        if chunk:
            chunk["step"] = step
            self._sendto(json.dumps(chunk).encode("utf-8"))

    def _json(self, value):
        value = self.ensure_data(value)
        if isinstance(value, np.ndarray):
            return value.item() if value.size == 1 else value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {k: self._json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json(v) for v in value]
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        try:
            return float(value)
        except Exception:
            return str(value)

    @registry.register(Scalar)
    def log_scalar(self, name: str, item: Scalar, step: int):
        self._pending[name] = self._json(item.value)

    @registry.register(Matrix)
    def log_matrix(self, name: str, item: Matrix, step: int):
        for key, value in self.matrix_items(name, item):
            self._pending[key] = value

    @registry.register(Text)
    def log_text(self, name: str, item: Text, step: int):
        self._pending[name] = item.text

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
            print("[PlotJugglerLogger] UDP socket closed.")


class AsyncLoggerWorker:
    def __init__(self, queue_: mp.Queue, backends: list[Logger]):
        self.queue = queue_
        self.backends = backends
        self.parent_pid = os.getppid()
        self.close_signal = False

    @classmethod
    def start(cls, queue_: mp.Queue, backends: list[Logger]):
        cls(queue_, backends).run()

    def run(self):
        threading.Thread(target=self.watchdog, daemon=True).start()
        while True:
            packet = self.queue.get()
            if packet is None:
                self.close_signal = True
                break
            record, step = packet
            self.log_to_backends(record, step)
        self.close_backends()

    def watchdog(self):
        while True:
            if self.close_signal:
                return
            try:
                os.kill(self.parent_pid, 0)
            except OSError:
                print(
                    f"\n\033[93m[AsyncLogger Watchdog] Parent process {self.parent_pid} "
                    "has been killed! Clean up and Exit\033[0m"
                )
                self.close_backends()
                os._exit(0)
            time.sleep(1.0)

    def log_to_backends(self, record: LogRecord, step: int):
        for backend in self.backends:
            try:
                backend.log(record, step)
            except Exception as exc:
                print(
                    f"\033[91m[AsyncLogger Error]: {type(backend).__name__} failed: "
                    f"{exc}\033[0m"
                )

    def close_backends(self):
        for backend in self.backends:
            try:
                backend.close()
            except Exception:
                pass


class AsyncLogger(Logger):
    def __init__(
        self,
        backends: list[Logger],
        queue_size: int = 1000,
        enable: bool = True,
        interval: int = 1,
        strict: bool = False,
    ):
        super().__init__(
            workspace=None, enable=enable, interval=interval, strict=strict
        )
        self.backends = backends
        self.queue_size = queue_size
        for logger in backends:
            logger.enable = enable
        self._process = None
        self._queue = None
        self._owner_pid = os.getpid()
        if mp.current_process().name == "MainProcess":
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.close)

    def init_impl(self):
        return None

    def _signal_handler(self, sig, frame):
        print(f"\n[AsyncLogger] Caught signal {sig}: flushing logger backends")
        self.close()
        if sig == signal.SIGINT:
            signal.signal(sig, signal.default_int_handler)
            raise KeyboardInterrupt
        signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)

    def _lazy_init(self):
        if os.getpid() != self._owner_pid:
            return
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=self.queue_size)
        self._process = ctx.Process(
            target=AsyncLoggerWorker.start,
            args=(self._queue, self.backends),
            daemon=True,
        )
        self._process.start()

    def log(self, metrics: Mapping[str, LogItem], step: int):
        if not self.enable or not self.is_active(step):
            return
        if os.getpid() != self._owner_pid:
            return
        record = LogRecord.from_mapping(metrics).activate_items(step)
        if not record:
            return
        if self.strict:
            self.validate_record(record)
        if self._process is None:
            self._lazy_init()
        if not self._process.is_alive():
            return
        try:
            self._queue.put_nowait((record, step))
        except queue.Full:
            pass

    @classmethod
    def validate_record(cls, record: LogRecord) -> None:
        for name, item in record.items():
            cls.validate_payload(item, name)

    @classmethod
    def validate_payload(cls, value, path: str) -> None:
        if getattr(value, "requires_grad", False):
            raise ValueError(
                f"AsyncLogger received {path} with requires_grad=True. "
                "detach metric payloads before logging."
            )
        if isinstance(value, LogItem):
            for name, _, _, _ in value.__class__.__attr_data_core_fields__:
                cls.validate_payload(getattr(value, name), f"{path}.{name}")
            for name, item in value._data.items():
                cls.validate_payload(item, f"{path}.{name}")
            return
        if isinstance(value, Mapping):
            for name, item in value.items():
                cls.validate_payload(item, f"{path}.{name}")
            return
        if isinstance(value, tuple):
            for index, item in enumerate(value):
                cls.validate_payload(item, f"{path}[{index}]")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                cls.validate_payload(item, f"{path}[{index}]")

    def close(self):
        if os.getpid() != self._owner_pid:
            return
        if self._process is not None and self._process.is_alive():
            try:
                if self._queue is not None:
                    self._queue.put(None)
                self._process.join(timeout=1.0)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=0.5)
            except Exception:
                pass
            finally:
                self._process = None


__all__ = [
    "Artifact",
    "AsyncLogger",
    "Figure",
    "Histogram",
    "Image",
    "LogItem",
    "LogItemRegistry",
    "LogRecord",
    "Logger",
    "LoggerPlugin",
    "PlotJugglerLogger",
    "Scalar",
    "SwanLabLogger",
    "TensorBoardLogger",
    "Text",
    "Matrix",
    "WandbLogger",
]
