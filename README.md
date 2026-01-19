# Thunder

A library for robot learning, which can also be used as an extension library for [pytorch](https://pytorch.org/), [jax](https://docs.jax.dev/en/latest/), [flax](https://flax.readthedocs.io/en/latest/) and [warp](https://nvidia.github.io/warp/)(Future Planning).

## Introduction

Thunder is a package that aims to simplify the process of creating and modifying algorithms by defining certain standards. In Thunder, we consider an algorithm to be a pipeline constructed from a sequence of atomic operations, each interacting with a context. A context contains all the data required by every operation within the pipeline. Each operation utilizes certain data from the context, computes various results, and then writes these back into the context. Consequently, for any given algorithm, we need only define a sequence of atomic operations. Should we wish to modify an algorithm, we need only replace one or several of these operations. With a diverse set of operators, we can, in principle, be able to construct algorithms with arbitrary topological structures.

## Core Concepts

### Executor && Module

An Executor refers to the backend implementation for a specific architecture (e.g., JAX, Torch). For instance, model definition, execution flow, and optimization processes differ entirely between the JAX and Torch backends. Therefore, we require a separately implemented Executor to provide a consistent API interface for common operations. The same applies to Modules. In JAX, when defining neural networks, we must inherit from `flax.nnx.Module`, whereas in PyTorch, we must inherit from `torch.nn.Module`. The differences between the two are substantial. For instance, when implementing algorithms using JAX, we must handle state management with extreme care. However, this is entirely unnecessary in the `PyTorch` backend. These differences necessitate defining Modules separately for each architecture to achieve consistency. Nevertheless, we want users to define neural networks freely according to their preferences. Therefore, instead of inheriting from our defined `Module`, we directly use the container `ModelPack` for encapsulation.

### Operation

There are all kinds of algorithms in this world. Take me, for instance—I work in reinforcement learning. Within reinforcement learning, there are numerous algorithms: `PPO`, `SAC`, `Dreamer`, `TD-MPC` and so on. There are too many to count. Whenever I have an idea and want to validate it quickly, the framework limits me. You see, I have to modify numerous upstream and downstream components just to implement the algorithm I envision. When I want to incorporate methods from other domains, the workload becomes even heavier and more challenging. If only we could write algorithms as easily as we draw algorithmic framework diagrams.

So let's consider what common characteristics algorithms share. After reflection, I believe algorithms consist of a series of building blocks. For example, the simplest `PPO` algorithm comprises:
- Interacting with the environment to collect data into the buffer.
- Sampling from the buffer to obtain a mini-batch.
- Calculating the loss.
- Updating the network.

In robotics, we often use `Recurrent PPO`, which requires modifying the structure:
- Remains unchanged.
- Remains unchanged.
- Splits the mini-batch into trajectories.
- Calculates the loss, adding a custom loss.
- Updates the network.

When we want to use `SAC`, many processes are similar, but the modifications become tedious and cumbersome. When transitioning to world models, significant architectural changes may be required to implement new algorithms. Do these algorithms share commonalities in each step? Each step appears to perform certain operation, store results, and pass them to subsequent operations. Thus, algorithms seem composed of a sequence of operations forming the desired outcome. In programming, we abstract each step into `Atomic Operations`—a common base class.

Next, we will introduce several special types of operations：

**`Objective`** is a special type of Operation whose purpose is to compute the loss according to the user-defined operation.

**`OptimizeOp`** Accept a series of objectives and perform gradient descent optimization on the network using the optimizer held internally by the algorithm.

**`Pipeline`** A special type of operation that internally holds a certain number of operations, executing them sequentially. This can also be understood as `JIT` technology accelerating internal pipeline operations.

These three are the most common special **`Operation`**.

### Algorithm

Finally, after all this work, we've built the `Algorithm`. The algorithm internally maintains a `Pipeline`, which must implement a `step` function. The `step` function executes the `Pipeline` once. The entire pipeline interacts with the algorithm's `ExecutionContext`, returning both the `ExecutionContext` and a dictionary. The new `ExecutionContext` contains the effects after all operations have completed, while the dictionary holds user-defined debug information.

For example, we want to implement a basic PPO algorithm.
- First, we need to define a model.
  ```python
  # For demonstration, we've simplified the creation process.
  models = ModelPack(actor=actor, critic=critic)
  ```
- Then we create the algorithm class and define the optimizer.
  ```python
  # For demonstration, we've simplified the creation process.
  algo = Algorithm(models)
  algo.build({"opt":{"targets": ["actor", "critic"], "class": "Adam ", "lr": 1e-4}})
  ```
- Then setup the pipeline
  ```python
  # For demonstration, we've simplified the creation process.
  algo.setup_pipeline([
    InteractOp(...),
    MiniBathTrain(..., pipeline=[
        SplitTraj(...),
        OptimizeOp("opt", [SurrogateLoss(),...])
        ])
    ])
  ```

## Welcome to Thunder Discussion

Although the concept is excellent—implementing various algorithms through composable operations—the sheer diversity of algorithms still demands concise designs for each operation. We strive to keep every operation's functionality and design as elegant as possible. I'm working hard on this, and of course, it wouldn't be possible without everyone's contributions. Therefore, we warmly welcome anyone to discuss the implementation of this repository with us: [Discussions](https://github.com/ecstayalive/thunder/discussions)
