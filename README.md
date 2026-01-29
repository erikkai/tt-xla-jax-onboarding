# Tenstorrent TT-XLA Onboarding and JAX Tutorials
This repository provides onboarding content for developers working with the Tenstorrent TT-XLA compiler front end. It bridges the gap between hardware configuration and executing optimized JAX models on physical silicon.

> **NOTE:** This is not official Tenstorrent documentation. This repo is a showcase of content created while working at Tenstorrent, and preserved here as a sample that shows writing and coding ability. The code sample for the tutorial [Compiling JAX Models with TT-XLA for Multi-Chip Execution Using Tensor Parallelism](https://github.com/erikkai/tt-xla-jax-onboarding/blob/main/tutorials/compile_multi_chip_w_tensor.md) was created collaboratively with another engineer. The other samples were created by me.

# How to Use This Repository 
This content is organized to reflect the developer journey from environment setup to advanced implementation.

## Environment Configuration
Depending on what you are doing, you will need a different getting started document: 
* [Getting Started](./setup/getting_started.md) - This is the easiest option and is best when you want to quickly start running models.
* [Getting Started with Docker](./setup/getting_started_docker.md) - If you are familiar with Docker, this is a great way to setup a consistent environment and run models.
* [Getting Started with Building from Source](./setup/getting_started_build_from_source.md) - This set up is the most difficult, and is only required if you want to help develop TT-XLA itself. 

## JAX Compilation Tutorials
After you configure your environment, follow these tutorials in order to learn how to work with TT-XLA and JAX: 
* [Compiling Models in PyTorch and JAX Using TT-XLA (Single Chip)](./tutorials/compile_models_single_chip.md) – Baseline execution on a single device.
* [Compiling JAX Models with TT-XLA for Multi-Chip Execution Using Data Parallelism](./tutorials/compile_models_multi_chip.md) – Distributing workloads across multiple chips.
* [Compiling JAX Models with TT-XLA for Multi-Chip Execution Using Tensor Parallelism](./tutorials/compile_multi_chip_w_tensor.md) – Advanced weight matrix sharding for large-scale inference.
