from __future__ import annotations

import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from thunder.core import DistributedContextManager, quiet_unless_main
from thunder.env import make_env, ThunderEnv
from thunder.rl.torch import SaveModels
from thunder.rl.torch.agent import PpoAgent
from thunder.utils import AsyncLogger, TensorBoardLogger
from thunder.utils.experiment import Experiment


def generate_step_panel(step, total_step, duration, metrics=None):
    info_table = Table(box=None, show_header=False, padding=(0, 2), width=40)
    info_table.add_row("[bold green]RunTime:[/]", f"[bold yellow]{duration:.4f}s[/]")
    return Panel(
        renderable=info_table,
        title=f"[magenta]Algorithm Iteration: {step} / {total_step}[/]",
        title_align="center",
        border_style="magenta",
        expand=False,
    )


def main():
    spec = Experiment.parse()
    with DistributedContextManager() as dist, quiet_unless_main(dist):
        Experiment.bind(spec, dist)
        env: ThunderEnv = make_env(spec.env)
        spec = Experiment.apply_task_cfg(spec)
        experiment = Experiment.start(spec, dist)
        agent, env = PpoAgent.factory(env, spec.agent)
        agent: PpoAgent

        if experiment.is_main:
            agent.pipeline.append(
                SaveModels(spec.save_interval, experiment.workspace, enable=spec.save)
            )
            print(agent)
            logger = AsyncLogger(
                [TensorBoardLogger(experiment.workspace)], enable=spec.log
            )
        else:
            logger = AsyncLogger([], enable=False)

        console = Console()
        for _ in range(spec.iteration):
            start = time.time()
            metrics = agent.step()
            duration = time.time() - start
            logger.log(metrics, agent.ctx.step)
            if experiment.is_main:
                console.print(
                    generate_step_panel(
                        agent.ctx.step, spec.iteration, duration, metrics
                    )
                )

        experiment.finish()


if __name__ == "__main__":
    main()
