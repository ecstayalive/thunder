# Thunder

<span class="dx-eyebrow">Thunder</span>

Thunder is a **general robot-learning library**, and also the algorithm package inside the DexLab MonoRepo — but it isn't tied to DexLab; any simulation platform that meets the standard interface can plug in (see the note below). Its core claim fits in one sentence: **treat an algorithm as a pipeline of atomic Operations, each reading from and writing to a shared blackboard (`ExecutionContext`).**

To change an algorithm, you swap one or two Operations in the pipeline; everything upstream and downstream stays put. This makes "writing an algorithm like drawing a block diagram" possible — which is exactly the pain Thunder targets: whenever you have a new idea, the framework shouldn't force you to rewrite a long chain of up- and downstream components.

!!! info "Not built only for DexLab"
    Thunder is a general robot-learning library and is **simulator-agnostic**: it ships loaders for IsaacLab, Gymnasium, DeepMind Control, ManiSkill, MuJoCo (mjlab), and more (one `EnvLoaderSpec` standard interface). DexLab is just one provider. See [Cooperation · Neither side is locked in](../concepts/data-flow.md).

!!! quote "The Thunder philosophy"
    - Seeking order in chaos, finding truth in phenomena
    - Maintain good performance while preserving elegance
    - Ideas are cheap, show me the mathematics and the code
    - Prefer duplicating code over a bad abstraction

## What it solves

Reinforcement learning has too many algorithms — PPO, SAC, Dreamer, TD-MPC… Yet pull any one apart and it is a sequence of "read data → compute something → write it back → pass it on." Thunder abstracts that step into an `Operation`, assembles a chain of them into a `Pipeline`, and lets an `Algorithm` repeatedly `step` it. So:

- **Build a new algorithm** = write a few new Operations and chain them;
- **Modify an algorithm** = swap `SplitTraj` or `ComputeGae` in the PPO pipeline for something else;
- **Multi-backend** = Operations never touch a concrete tensor library; an `Executor` / `Module` hides the torch-vs-jax differences.

## Section map

<div class="grid cards" markdown>

-   :material-lightning-bolt-outline:{ .lg .middle } &nbsp;**Philosophy**

    ---

    Why treat an algorithm as an operation pipeline? Executor / Module / ModelPack, and `ExecutionContext` as the blackboard.

    [:octicons-arrow-right-24: Philosophy](philosophy.en.md)

-   :material-vector-polyline:{ .lg .middle } &nbsp;**Operations & Pipeline**

    ---

    The Operation abstraction, plus the three special operators Objective / OptimizeOp / Pipeline. Compose and replace like building blocks.

    [:octicons-arrow-right-24: Operations & Pipeline](operations.en.md)

-   :material-robot-outline:{ .lg .middle } &nbsp;**Agents & algorithms**

    ---

    Agent = models + strategy + pipeline; how `obs_keys` aligns environment observations to the network backbone.

    [:octicons-arrow-right-24: Agents & algorithms](agents.en.md)

-   :material-graph-outline:{ .lg .middle } &nbsp;**Models & Networks**

    ---

    What models exist, the `ModelSpec.factory` construction pattern, and how actor / critic assemble backbones + distribution heads.

    [:octicons-arrow-right-24: Models & Networks](models.en.md)

-   :material-chart-timeline-variant:{ .lg .middle } &nbsp;**PPO in practice**

    ---

    The real collect → GAE → loss → optimize pipeline; Recurrent PPO, ScaledBeta bounded actions, and how to run it.

    [:octicons-arrow-right-24: PPO in practice](ppo.en.md)

</div>

!!! info "Active backend"
    Thunder plans to support PyTorch / JAX(flax) / Warp backends. **The active backend is torch** (`thunder.rl.torch`, which DexLab also builds on); the JAX path is already wired in at key points such as `ExecutionContext` pytree registration, but the RL operators live mainly on the torch side. Every example in this section uses the torch backend.

---

Start with the philosophy to internalize the "algorithm-as-pipeline" worldview: [Philosophy →](philosophy.en.md)
