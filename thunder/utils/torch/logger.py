import atexit
import datetime
import os
import pathlib
import signal
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import coolname
import torch
import torch.multiprocessing as mp


class Workspace:
    """
    Args:
        ...
    """

    def __init__(self, root: str, project: str, run_name: str = None, timestamp: bool = False):
        self.root = pathlib.Path(root)
        self.project = project
        self.run_name = run_name if run_name else coolname.generate_slug(2)
        if timestamp:
            time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.run_name = f"{time_stamp}-{self.run_name}"
        self.run_dir = self.root / project / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"Workspace(run_dir={self.run_dir})"


class Logger(ABC):
    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self._logger = None

    def _ensure(self):
        if self._logger is None:
            self.init()

    def log(self, metrics: Dict[str, Any], step: int):
        self._ensure()
        self.log_impl(metrics, step)

    @abstractmethod
    def init(self):
        """ """
        raise NotImplementedError

    @abstractmethod
    def log_impl(self, metrics: Dict[str, Any], step: int):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class TensorBoardLogger(Logger):

    def init(self):
        from torch.utils.tensorboard import SummaryWriter

        self._logger = SummaryWriter(log_dir=self.workspace.run_dir)

    def log_impl(self, metrics: Dict[str, torch.Tensor], step: int):
        for k, v in metrics.items():
            self._logger.add_scalar(k, v, global_step=step)

    def close(self):
        if self._logger is not None:
            self._logger.close()


class WandbLogger(Logger):
    def __init__(self, workspace: Workspace, config: Dict = None, **kwargs):
        super().__init__(workspace)
        self.config = config
        self.kwargs = kwargs

    def init(self):
        import wandb

        self._logger = wandb.init(
            project=self.workspace.project,
            name=self.workspace.run_name,
            dir=self.workspace.run_dir,
            config=self.config,
            resume="allow",
            reinit=True,
            **self.kwargs,
        )

    def log_impl(self, metrics: Dict[str, Any], step: int):
        self._logger.log(metrics, step=step)

    def close(self):
        if self._logger is not None:
            self._logger.finish()


class SwanLabLogger(Logger):
    def __init__(self):
        super().__init__()

    def init(self):
        import swanlab

    def log_impl(self, metrics, step):
        return super().log_impl(metrics, step)

    def close(self):
        if self._logger is not None:
            pass


def _logger_worker_entry(queue: mp.Queue, backends: List[Logger]):
    while True:
        packet = queue.get()
        if packet is None:
            break
        raw_metrics, step = packet
        cpu_metrics = {k: (v.item() if hasattr(v, "item") else v) for k, v in raw_metrics.items()}
        for b in backends:
            try:
                b.log(cpu_metrics, step)
            except Exception as e:
                print(f"\033[91m[AsyncLogger Error]: {type(b).__name__} failed: {e}\033[0m")
    for b in backends:
        b.close()


class AsyncLogger:
    """_summary_"""

    def __init__(self, backends: List[Logger], queue_size: int = 1000):
        self.backends = backends
        self.queue_size = queue_size
        self._process = None
        self._queue = None
        self._owner_pid = os.getpid()
        if mp.current_process().name == "MainProcess":
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.close)

    def _signal_handler(self, sig, frame):
        print(f"\n[AsyncLogger] Detect KeyboardInterrupt: Clean Up Resources and Exit")
        self.close()
        sys.exit(0)

    def _lazy_init(self):
        if os.getpid() != self._owner_pid:
            return
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=self.queue_size)
        self._process = ctx.Process(
            target=_logger_worker_entry, args=(self._queue, self.backends), daemon=True
        )
        self._process.start()

    def log(self, metrics: Dict[str, Any], step: int):
        if os.getpid() != self._owner_pid:
            return
        if self._process is None:
            self._lazy_init()
        if not self._process.is_alive():
            return
        safe_metrics = {
            k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()
        }
        try:
            self._queue.put_nowait((safe_metrics, step))
        except mp.queues.Full:
            pass

    def close(self):
        if os.getpid() != self._owner_pid:
            return
        if self._process is not None and self._process.is_alive():
            try:
                if self._queue is not None:
                    self._queue.put(None)
                self._process.join(timeout=0.2)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=0.1)
            except Exception:
                pass
            finally:
                self._process = None
                self._process = None
                self._process = None
                self._process = None
                self._process = None
