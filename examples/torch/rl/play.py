from __future__ import annotations

import sys
import time

from thunder.env import make_env, ThunderEnv
from thunder.rl.torch.agent import Play, PpoAgent
from thunder.utils import ArgParser, PlotJugglerLogger
from thunder.utils.experiment import Experiment, ExperimentSpec, PlaySpec
from thunder.utils.workspace import Workspace


def sleep_until(target_time: float):
    """Sleep until target_time (monotonic) with sub-millisecond precision."""
    while True:
        remaining = target_time - time.monotonic()
        if remaining <= 0:
            break
        if remaining > 0.001:
            time.sleep(remaining - 0.001)
        else:
            pass


def main():
    argv = sys.argv[1:]
    args = ArgParser(PlaySpec).parse()

    run_dir = args.locate_run()
    workspace = Workspace.open(run_dir)
    spec = workspace.load_config(ExperimentSpec)
    spec = ArgParser(type(spec), default_instance=spec, overlay=True).parse(argv)

    which = int(args.checkpoint) if args.checkpoint.isdigit() else args.checkpoint
    checkpoint = workspace.resolve_checkpoint(which)
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"No '{args.checkpoint}' checkpoint in {run_dir}")

    spec.agent.resume = str(checkpoint)
    spec.agent.compile = False
    logger = PlotJugglerLogger(workspace=workspace, enable=spec.log)
    Experiment.bind(spec)
    print(
        f"[play] run={run_dir.name}  task={spec.env.task}  checkpoint={checkpoint.name}"
    )

    env: ThunderEnv = make_env(spec.env)
    agent, env = PpoAgent.factory(env, spec.agent)
    agent: PpoAgent
    agent.setup_pipeline((Play(env, agent, explore=args.explore),))
    print(agent)

    step_dt = getattr(env.unwrapped, "step_dt", 1.0 / 60.0)
    interval = step_dt / args.speed
    next_step_time = time.monotonic()
    for _ in range(args.iteration):
        metrics = agent.step()
        logger.log(metrics, step=agent.ctx.step)
        next_step_time += interval
        sleep_until(next_step_time)
        if next_step_time < time.monotonic():
            next_step_time = time.monotonic()


if __name__ == "__main__":
    main()
