from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch

VIEW_REGISTRY = {}


def register_view(name):
    def decorator(cls):
        VIEW_REGISTRY[name] = cls
        return cls

    return decorator


class TSNEView(ABC):
    """ """

    def __init__(self, independent: bool = False, **kwargs):
        self.config = kwargs
        self.independent = independent

    @abstractmethod
    def draw(self, ax: plt.Axes, offsets: torch.Tensor, coords_2d: np.ndarray, step: int, **kwargs):
        """
        Draws the plot on the provided Axes.
        """
        raise NotImplementedError


@register_view("traj")
class TrajectoryView(TSNEView):
    def __init__(
        self,
        independent: bool = False,
        plot_nums: int = 16,
        plot_connection: bool = False,
        seed=0,
        **kwargs,
    ):
        super().__init__(independent, **kwargs)
        self.plot_nums = plot_nums
        self.plot_connection = plot_connection
        torch.manual_seed(seed)

    def draw(self, ax: plt.Axes, offsets, coords_2d, step, trace_indices=None):
        cmap = plt.get_cmap("gist_rainbow")
        num_trajs = offsets.size(0) - 1
        num_plot = min(num_trajs, self.plot_nums)
        if trace_indices is not None and not self.independent:
            limit = min(len(trace_indices), self.plot_nums)
            plot_indices = trace_indices[:limit].tolist()
        else:
            plot_indices = torch.randperm(num_trajs)[:num_plot].tolist()
        for i, idx in enumerate(plot_indices):
            traj_color = cmap(i / max(num_plot - 1, 1))
            start = offsets[idx].cpu().numpy()
            end = offsets[idx + 1].cpu().numpy()
            points = coords_2d[start:end]

            if self.plot_connection and len(points) > 1:
                self._draw_arrows(ax, points, traj_color)

            ax.scatter(points[:, 0], points[:, 1], color=[traj_color], s=32, alpha=0.8)
            # Add Start/End Labels
            ax.text(
                points[0, 0], points[0, 1], f"S{i}", color=traj_color, fontsize=10, weight="bold"
            )
            ax.text(
                points[-1, 0], points[-1, 1], f"E{i}", color=traj_color, fontsize=10, weight="bold"
            )
        if self.plot_connection:
            ax.set_title(f"Trajectories [{num_plot}/{num_trajs}] | Step {step}")
        else:
            ax.set_title(f"Points in trajectories [{num_plot}/{num_trajs}] | Step {step}")

    def _draw_arrows(self, ax, points, color):
        x, y = points[:-1, 0], points[:-1, 1]
        u, v = points[1:, 0] - x, points[1:, 1] - y
        ax.quiver(
            x,
            y,
            u,
            v,
            color=color,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            headwidth=3,
            headlength=5,
            headaxislength=4,
            minshaft=1.5,
            minlength=0.5,
            alpha=0.8,
            zorder=2,
            linewidth=0.1,
            antialiased=True,
        )


@register_view("phase")
class TrajPhaseView(TSNEView):
    """Draws only Start, Middle, and End points."""

    def __init__(
        self, independent: bool = False, plot_nums=128, plot_connection: bool = False, seed=0
    ):
        super().__init__(independent=independent)
        self.plot_nums = plot_nums
        self.plot_connection = plot_connection
        torch.manual_seed(seed)

    def draw(self, ax: plt.Axes, offsets, coords_2d, step, trace_indices=None):
        num_trajs = offsets.size(0) - 1
        num_plot = min(num_trajs, self.plot_nums)
        if trace_indices is not None and not self.independent:
            limit = min(len(trace_indices), self.plot_nums)
            plot_indices = trace_indices[:limit].tolist()
        else:
            plot_indices = torch.randperm(num_trajs)[:num_plot].tolist()

        for i, idx in enumerate(plot_indices):
            start = offsets[idx].cpu().numpy()
            end = offsets[idx + 1].cpu().numpy()
            p_start = coords_2d[start]
            p_end = coords_2d[end - 1]
            p_mid = coords_2d[(start + end) // 2]
            if self.plot_connection:
                self._draw_quiver(ax, p_start, p_mid, "tab:blue")
                self._draw_quiver(ax, p_mid, p_end, "tab:blue")
            ax.scatter(*p_start, c="tab:green", s=40, alpha=0.8, label="Start" if i == 0 else None)
            ax.scatter(*p_end, c="tab:red", s=40, alpha=0.8, label="End" if i == 0 else None)
            ax.scatter(*p_mid, c="tab:orange", s=30, alpha=0.8, label="Mid" if i == 0 else None)

        if self.plot_connection:
            ax.set_title(f"Trajectory Phase [{num_plot}/{num_trajs}] | Step {step}")
        else:
            ax.set_title(f"Points in phase [{num_plot}/{num_trajs}] | Step {step}")

        ax.legend(loc="upper right", fontsize="x-small")

    def _draw_quiver(self, ax, p_start, p_end, color):
        ax.quiver(
            p_start[0],
            p_start[1],
            p_end[0] - p_start[0],
            p_end[1] - p_start[1],
            color=color,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            headwidth=3,
            headlength=5,
            headaxislength=4,
            minshaft=1.5,
            minlength=0.5,
            alpha=0.8,
            zorder=2,
            linewidth=0.1,
            antialiased=True,
        )
