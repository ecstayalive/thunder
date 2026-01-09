# Thunder

A library for robot learning, which can also be used as an extension library for [pytorch](https://pytorch.org/), [jax](https://docs.jax.dev/en/latest/) and [flax](https://flax.readthedocs.io/en/latest/).

## Introduction

Thunder is a package that aims to simplify the process of creating and modifying algorithms by defining certain standards. In Thunder, we consider an algorithm to be a pipeline constructed from a sequence of atomic operations, each interacting with a context. A context contains all the data required by every operation within the pipeline. Each operation utilizes certain data from the context, computes various results, and then writes these back into the context. Consequently, for any given algorithm, we need only define a sequence of atomic operations. Should we wish to modify an algorithm, we need only replace one or several of these operations. With a diverse set of operators, we can, in principle, be able to construct algorithms with arbitrary topological structures.

## Core Concepts

### Executor

### Module

### Operation

### Objective

### Algorithm
