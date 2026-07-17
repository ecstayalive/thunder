# Thunder

<span class="dx-eyebrow">Thunder</span>

Thunder 是一个**通用的机器人学习库**，也是 DexLab MonoRepo 里的算法包 —— 但它并不绑定 DexLab，任何符合标准接口的仿真平台都能接入（见下方说明）。它的核心主张只有一句话：**把一个算法看成一条由原子算子（Operation）组成的流水线，每个算子围绕一块共享黑板（`ExecutionContext`）读写数据。**

换算法，就是替换流水线里的一两个算子；上下游不动。这让"像画框图一样写算法"成为可能 —— 这正是 Thunder 想解决的痛点：每当你有一个新想法，不该被框架逼着去改一大串上下游组件。

!!! info "不只为 DexLab 而生"
    Thunder 是通用的机器人学习库，对仿真器**无关**：内置 IsaacLab、Gymnasium、DeepMind Control、ManiSkill、MuJoCo(mjlab) 等多平台 loader（统一的 `EnvLoaderSpec` 标准接口），DexLab 只是其中一个 provider。详见 [三包协作 · 两个方向都不绑死](../concepts/data-flow.md)。

!!! quote "Thunder 哲学"
    - 在混沌中寻找秩序，在现象中发现真理
    - 保持优雅的同时保持性能
    - Ideas are cheap，show me the mathematics and the code
    - 宁可复制代码，也不要一个糟糕的抽象

## 它解决什么

强化学习里算法太多了 —— PPO、SAC、Dreamer、TD-MPC…… 但拆开看，每个算法都是一串"取数据 → 算东西 → 写回去 → 传给下一步"。Thunder 把这一步抽象成 `Operation`，把一串 Operation 组装成 `Pipeline`，让 `Algorithm` 反复 `step` 它。于是：

- **造新算法** = 写几个新 Operation，串成 Pipeline；
- **改算法** = 把 PPO 流水线里的 `SplitTraj`、`ComputeGae` 换成别的；
- **多后端** = Operation 不碰具体张量库，由 `Executor` / `Module` 屏蔽 torch 与 jax 的差异。

## 本节导航

<div class="grid cards" markdown>

-   :material-lightning-bolt-outline:{ .lg .middle } &nbsp;**设计哲学**

    ---

    为什么把算法看作算子流水线？Executor / Module / ModelPack，以及作为黑板的 `ExecutionContext`。

    [:octicons-arrow-right-24: 设计哲学](philosophy.md)

-   :material-vector-polyline:{ .lg .middle } &nbsp;**Operation 与 Pipeline**

    ---

    Operation 抽象，以及三种特殊算子 Objective / OptimizeOp / Pipeline。如何像搭积木一样组合、替换。

    [:octicons-arrow-right-24: Operation 与 Pipeline](operations.md)

-   :material-robot-outline:{ .lg .middle } &nbsp;**Agent 与算法**

    ---

    Agent = 模型 + 策略 + pipeline；`obs_keys` 如何把环境观测对齐到网络骨干。

    [:octicons-arrow-right-24: Agent 与算法](agents.md)

-   :material-graph-outline:{ .lg .middle } &nbsp;**模型与网络**

    ---

    现有哪些模型、`ModelSpec.factory` 构造范式，以及 actor / critic 如何组装骨干 + 分布头。

    [:octicons-arrow-right-24: 模型与网络](models.md)

-   :material-chart-timeline-variant:{ .lg .middle } &nbsp;**PPO 实战**

    ---

    collect → GAE → loss → optimize 的真实流水线；Recurrent PPO、ScaledBeta 有界动作、怎么跑起来。

    [:octicons-arrow-right-24: PPO 实战](ppo.md)

</div>

!!! info "现役后端"
    Thunder 规划支持 PyTorch / JAX(flax) / Warp 多后端。**当前现役的是 torch 后端**（`thunder.rl.torch`，DexLab 也基于它）；JAX 路径在 `ExecutionContext` 的 pytree 注册等关键处已经预留，但 RL 算子主要在 torch 侧实现。本节示例一律以 torch 后端为准。

---

先从设计哲学开始，理解"算法即流水线"这套世界观：[设计哲学 →](philosophy.md)
