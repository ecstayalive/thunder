from __future__ import annotations

import math
from typing import Any

import torch

from thunder.core.data import attr_dataclass
from thunder.core.executor import Executor

from .workspace import Workspace
from .logging import Figure, LogItem, LoggerPlugin, LogItemRegistry


DEFAULT_TSNE_VIEWS = (
    "traj",
    ("traj", {"plot_connection": True}),
    "phase",
    ("phase", {"plot_connection": True}),
)


@attr_dataclass(slots=True)
class TSNE(LogItem):
    embedding: Any = None
    mask: Any = None
    views: Any = None
    perplexity: int = 30
    max_iter: int = 1000
    save: bool = True


class CuTSNEPlugin(LoggerPlugin):
    """Render TSNE LogItems into Figure LogItems.

    The plugin does not inspect ExecutionContext. Producers must explicitly
    provide the embedding and mask through a TSNE LogItem.
    """

    registry = LogItemRegistry()

    def __init__(
        self,
        workspace: Workspace,
        views: Any = DEFAULT_TSNE_VIEWS,
        enable: bool = True,
        interval: int = 1,
        strict: bool = False,
    ):
        super().__init__(enable=enable, interval=interval, strict=strict)
        self.workspace = workspace
        self.views_config = views
        self._views = None
        self._cols = None
        self._rows = None
        self._tsne = None
        self._tsne_config = None
        self._fig = None
        self._axes = None

    @registry.register(TSNE)
    def render_tsne(self, name: str, item: TSNE, step: int):
        if item.embedding is None or item.mask is None:
            return {}
        self._ensure_ready(item)

        embedding = Executor.to_torch(item.embedding)
        mask = Executor.to_torch(item.mask)
        mask_bool = mask.bool()
        traj_lengths = mask_bool.sum(dim=1)
        keep_indices = traj_lengths > 1
        valid_mask = mask_bool & keep_indices.unsqueeze(1)
        valid_embeddings = embedding[valid_mask]
        if valid_embeddings.numel() == 0:
            return {}

        valid_lengths: torch.Tensor = traj_lengths[keep_indices]
        offsets = torch.zeros(
            valid_lengths.size(0) + 1,
            dtype=torch.long,
            device=valid_lengths.device,
        )
        torch.cumsum(valid_lengths, dim=0, out=offsets[1:])
        try:
            coords_2d = self._tsne.fit_transform(valid_embeddings)
        except Exception:
            return {}

        num_trajs = offsets.size(0) - 1
        master_perm = torch.randperm(num_trajs, device=offsets.device)
        self._render(offsets, coords_2d, step, master_perm)
        if item.save:
            save_dir = self.workspace.run_dir / "tsne"
            save_dir.mkdir(parents=True, exist_ok=True)
            self._fig.savefig(save_dir / f"tsne-{step}.pdf")
        return {name: Figure(self._fig, close=False)}

    def _ensure_ready(self, item: TSNE):
        self._ensure_views(item.views or self.views_config)
        self._ensure_tsne(item.perplexity, item.max_iter)
        self._ensure_figure()

    def _ensure_views(self, views_config):
        if self._views is not None and views_config == self.views_config:
            return
        from .tsne_views import TSNEView, VIEW_REGISTRY

        views = []
        for view in views_config:
            if isinstance(view, str):
                if view not in VIEW_REGISTRY:
                    raise ValueError(f"Unknown view: {view}")
                views.append(VIEW_REGISTRY[view]())
            elif isinstance(view, (tuple, list)) and len(view) == 2:
                name, config = view
                if name not in VIEW_REGISTRY:
                    raise ValueError(f"Unknown view: {name}")
                views.append(VIEW_REGISTRY[name](**config))
            elif isinstance(view, TSNEView):
                views.append(view)
            else:
                raise ValueError(f"Invalid view format: {view}")
        self.views_config = views_config
        self._views = views
        n = len(self._views)
        self._cols = math.ceil(math.sqrt(n))
        self._rows = math.ceil(n / self._cols)
        self._fig = None
        self._axes = None

    def _ensure_tsne(self, perplexity: int, max_iter: int):
        config = (perplexity, max_iter)
        if self._tsne is not None and self._tsne_config == config:
            return
        from cuml.manifold import TSNE as CumlTSNE

        self._tsne = CumlTSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=max_iter,
            output_type="numpy",
        )
        self._tsne_config = config

    def _ensure_figure(self):
        if self._fig is not None:
            return
        import matplotlib.pyplot as plt
        import numpy as np

        plt.ion()
        self._fig, self._axes = plt.subplots(
            num="Latent Visualizer (T-SNE Logger)",
            figsize=(6 * self._cols, 5 * self._rows),
            nrows=self._rows,
            ncols=self._cols,
            tight_layout=True,
        )
        self._fig.suptitle(t="Latent Visualizer (T-SNE Logger)", fontweight="bold")
        if isinstance(self._axes, np.ndarray):
            self._axes = self._axes.flatten()
        else:
            self._axes = [self._axes]

    def _render(self, offsets, coords_2d, step, master_perm):
        import matplotlib.pyplot as plt

        for ax in self._axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        for idx, view in enumerate(self._views):
            view.draw(self._axes[idx], offsets, coords_2d, step, master_perm)
        for idx in range(len(self._views), len(self._axes)):
            self._axes[idx].axis("off")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        if self._fig is None:
            return
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.close(self._fig)
        self._fig = None
        self._axes = None


__all__ = ["CuTSNEPlugin", "DEFAULT_TSNE_VIEWS", "TSNE"]
