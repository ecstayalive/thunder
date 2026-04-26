from __future__ import annotations

import atexit
import math
import os
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union, final

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils._pytree as pytree

from thunder.core import Ref

from .tsne_views import VIEW_REGISTRY, TSNEView
from .workspace import Workspace

if TYPE_CHECKING:
    from thunder.core import ExecutionContext


class Logger(ABC):
    def __init__(self, workspace: Workspace, enable: bool = True):
        self.workspace = workspace
        self.initialized = False
        self.enable = enable

    @final
    def log(self, metrics: Dict[str, Any], step: int):
        if not self.enable:
            return
        if not self.initialized:
            self.init()
        self.log_impl(metrics, step)

    @final
    def init(self):
        """ """
        if not self.workspace.run_dir.exists():
            self.workspace.mkdir()
        self.init_impl()
        self.initialized = True

    @abstractmethod
    def init_impl(self):
        raise NotImplementedError

    @abstractmethod
    def log_impl(self, metrics: Dict[str, Any], step: int):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class TensorBoardLogger(Logger):

    def init_impl(self):
        from torch.utils.tensorboard import SummaryWriter

        self._logger = SummaryWriter(log_dir=self.workspace.run_dir)

    def log_impl(self, metrics: Dict[str, torch.Tensor], step: int):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                self._logger.add_scalar(k, v.cpu().item(), global_step=step)
            elif isinstance(v, (int, float)):
                self._logger.add_scalar(k, v, global_step=step)
            elif isinstance(v, dict):
                self.log_impl({f"{k}/{sub_k}": sub_v for sub_k, sub_v in v.items()}, step)

    def close(self):
        if self._logger is not None:
            self._logger.close()


class WandbLogger(Logger):
    def __init__(self, workspace: Workspace, config: Dict = None, **kwargs):
        super().__init__(workspace)
        self.config = config
        self.kwargs = kwargs

    def init_impl(self):
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
        metrics = {
            k: (v.cpu().item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()
        }
        self._logger.log(metrics, step=step)

    def close(self):
        if self._logger is not None:
            self._logger.finish()


class SwanLabLogger(Logger):
    def __init__(self):
        super().__init__()

    def init_impl(self):
        import swanlab

    def log_impl(self, metrics: Dict[str, Any], step: int):
        pass

    def close(self):
        pass


class RerunLogger(Logger):
    def __init__(self):
        super().__init__()

    def init_impl(self):
        import rerun as rr

    def log_impl(self, metrics: Dict[str, Any], step: int):
        pass

    def close(self):
        pass


class CuTSNELogger(Logger):
    def __init__(
        self,
        workspace: Workspace,
        target_ref: Ref = Ref("batch.embedding"),
        views: List[Union[str, Tuple[str, Dict], TSNEView]] = [
            "traj",
            ("traj", {"plot_connection": True}),
            "phase",
            ("phase", {"plot_connection": True}),
        ],
        perplexity: int = 30,
        max_iter: int = 1000,
        interval: int = 20,
        enable: bool = True,
    ):
        super().__init__(workspace, enable=enable)
        self.target_ref = target_ref
        # TSNE config
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.interval = interval

        self.views = []
        for v in views:
            self.views.append(self._create_view(v))
        # Auto Layout
        n = len(self.views)
        self.cols = math.ceil(math.sqrt(n))
        self.rows = math.ceil(n / self.cols)

        self._tsne = None
        self._fig = None
        self._axes = None
        self.save_dir = self.workspace.run_dir / "tsne"

    def _create_view(self, v):
        if isinstance(v, str):
            if v not in VIEW_REGISTRY:
                raise ValueError(f"Unknown view: {v}")
            return VIEW_REGISTRY[v]()
        # Examples: Tuple -> ("traj", {"plot_nums": 10})
        elif isinstance(v, (tuple, list)) and len(v) == 2:
            name, config = v
            if name not in VIEW_REGISTRY:
                raise ValueError(f"Unknown view: {name}")
            return VIEW_REGISTRY[name](**config)
        elif isinstance(v, TSNEView):
            return v
        else:
            raise ValueError(f"Invalid view format: {v}")

    def init_impl(self):
        if self.enable:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        from cuml.manifold import TSNE as cumlTSNE

        self._tsne = cumlTSNE(
            n_components=2, perplexity=self.perplexity, max_iter=self.max_iter, output_type="numpy"
        )

        plt.ion()
        self._fig, self._axes = plt.subplots(
            num="Latent Visualizer (T-SNE Logger)",
            figsize=(6 * self.cols, 5 * self.rows),
            nrows=self.rows,
            ncols=self.cols,
            tight_layout=True,
        )
        self._fig.suptitle(t="Latent Visualizer (T-SNE Logger)", fontweight="bold")
        if isinstance(self._axes, np.ndarray):
            self._axes = self._axes.flatten()
        else:
            self._axes = [self._axes]

    def log_impl(self, metrics: Dict[str, Any], step: int):
        ctx: ExecutionContext = metrics["execution_context"]
        embedding = self.target_ref(ctx)
        mask = ctx.batch.mask
        if step % self.interval != 0:
            return
        if embedding is None or mask is None:
            return
        mask_bool = mask.bool()
        traj_lengths = mask_bool.sum(dim=1)
        keep_indices = traj_lengths > 1
        valid_mask = mask_bool & keep_indices.unsqueeze(1)
        valid_embeddings = embedding[valid_mask]
        valid_lengths: torch.Tensor = traj_lengths[keep_indices]
        offsets = torch.zeros(
            valid_lengths.size(0) + 1, dtype=torch.long, device=valid_lengths.device
        )
        torch.cumsum(valid_lengths, dim=0, out=offsets[1:])
        try:
            coords_2d = self._tsne.fit_transform(valid_embeddings)
        except Exception as e:
            return
        num_trajs = offsets.size(0) - 1
        master_perm = torch.randperm(num_trajs, device=offsets.device)
        self._render(offsets, coords_2d, step, master_perm)

    def _render(self, offsets, coords_2d, step, master_perm):
        for ax in self._axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        for i, view in enumerate(self.views):
            view.draw(self._axes[i], offsets, coords_2d, step, master_perm)
        # Hide unused subplots
        for i in range(len(self.views), len(self._axes)):
            self._axes[i].axis("off")
        if self.enable:
            plt.savefig(f"{self.save_dir}/tsne-{step}.pdf")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self._fig)


def _logger_worker_entry(queue: mp.Queue, backends: List[Logger]):
    parent_pid = os.getppid()
    _close_signal = False

    def watchdog():
        """ """
        while True:
            if _close_signal:
                return
            try:
                os.kill(parent_pid, 0)
            except OSError:
                print(
                    f"\n\033[93m[AsyncLogger Watchdog] Parent process {parent_pid} has been killed! Clean up and Exit\033[0m"
                )
                for b in backends:
                    try:
                        b.close()
                    except:
                        pass
                os._exit(0)
            time.sleep(1.0)

    monitor_thread = threading.Thread(target=watchdog, daemon=True)
    monitor_thread.start()

    while True:
        packet = queue.get()
        if packet is None:
            _close_signal = True
            break
        safe_metrics, step = packet
        for b in backends:
            try:
                b.log(safe_metrics, step)
            except Exception as e:
                print(f"\033[91m[AsyncLogger Error]: {type(b).__name__} failed: {e}\033[0m")
    for b in backends:
        b.close()


class AsyncLogger:
    """ """

    def __init__(self, backends: List[Logger], queue_size: int = 1000, enable: bool = True):
        self.backends = backends
        self.queue_size = queue_size
        self.enable = enable
        for logger in backends:
            logger.enable = enable
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
        if not self.enable:
            return
        if os.getpid() != self._owner_pid:
            return
        if self._process is None:
            self._lazy_init()
        if not self._process.is_alive():
            return

        def _detach_all(leaf):
            if isinstance(leaf, torch.Tensor):
                return leaf.detach()
            else:
                return leaf

        safe_metrics = pytree.tree_map(_detach_all, metrics)

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
                self._process.join(timeout=0.1)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=0.1)
            except Exception:
                pass
            finally:
                self._process = None
