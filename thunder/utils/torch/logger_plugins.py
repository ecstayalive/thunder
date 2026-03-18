from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

if TYPE_CHECKING:
    from thunder.core import ExecutionContext

import math

from .tsne_views import *
from .workspace import Workspace

LOGGER_PLUGIN_REGISTRY = {}


def register_logger_plugin(name: str):
    def decorator(cls):
        LOGGER_PLUGIN_REGISTRY[name] = cls
        return cls

    return decorator


class LoggerPlugin(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, *args, **kwds):
        raise NotImplementedError


@register_logger_plugin("cutsne")
class CuTSNEPlugin(LoggerPlugin):
    def __init__(
        self,
        workspace: Workspace,
        name: str = "CuTsne",
        views: List[Union[str, Tuple[str, Dict], TSNEView]] = [
            "traj",
            ("traj", {"plot_connection": True}),
            "phase",
            ("phase", {"plot_connection": True}),
        ],
        perplexity: int = 30,
        max_iter: int = 1000,
        interval: int = 20,
        save: bool = True,
    ):
        super().__init__(name)
        self.workspace = workspace
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
        embeddings = ctx.batch["embeddings"]
        mask = ctx.batch.mask
        if step % self.interval != 0:
            return
        if embeddings is None or mask is None:
            return
        mask_bool = mask.bool()
        traj_lengths = mask_bool.sum(dim=1)
        keep_indices = traj_lengths > 1
        valid_mask = mask_bool & keep_indices.unsqueeze(1)
        valid_embeddings = embeddings[valid_mask]
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
        if self.save:
            plt.savefig(f"{self.save_dir}/tsne-{step}.pdf")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self._fig)
