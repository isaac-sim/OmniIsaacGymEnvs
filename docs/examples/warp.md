## Using Warp for Reinforcement Learning

[Warp](https://github.com/NVIDIA/warp) is a Python framework designed for high-performance GPU-accelerated programming.
Warp takes regular Python functions and JIT compiles them to efficient kernel code that can run on the CPU or GPU.
This can be useful in reinforcement learning when tasks are required to perform complex computations that can
benefit from being accelerated through C++/CUDA code, while implementing purely in Python.

In OmniIsaacGymEnvs, we provide a dedicated base class [RLTaskWarp](../../omniisaacgymenvs/tasks/base/rl_task.py), which
maintains the observation, reward, reset, and progress buffers in the form of [Warp arrays(https://nvidia.github.io/warp/_build/html/basics.html#arrays). This allows us to develop 
task logic using the Warp framework, replacing the use of PyTorch tensors and functions. Reward and observation
functions can be implemented as [Warp kernels](https://nvidia.github.io/warp/_build/html/basics.html#kernels), which are JIT-compiled to native CUDA code to take advantage of parallel acceleration on the GPU.

Examples implemented in Warp can be found at [tasks/warp](../../omniisaacgymenvs/tasks/warp). We provide three simple examples for reference: Cartpole, Ant and Humanoid. Although these examples may not show the full potential of the acceleration that can be achieved
by Warp due to their simplicity in nature, they can be a good reference point for implementing more complex tasks in Warp.