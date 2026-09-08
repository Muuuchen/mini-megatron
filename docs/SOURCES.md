# 原始资料与 Megatron 源码对应关系

检索/核对日期：2026-09-08。下面是阅读入口，不是本仓库运行依赖。Megatron 的 `main` 和文档 `latest` 会继续变化；本实验不承诺与某个 Megatron release API 兼容。

## 项目来源与风格参照

[S1] [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)：轻量推理引擎的参照。借鉴“沿着一条可运行路径理解系统”的学习方式，不复用其推理调度作为训练调度。

[S2] [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)：紧凑的 serving 系统实现。它与 nano-vllm 是这里所说 mini/nano 路线的具体例子，不声称所有轻量版本都有同一名称或官方归属。

[S3] [NVIDIA/Megatron-LM README](https://github.com/NVIDIA/Megatron-LM)：Megatron-LM 与 Megatron Core 的分层、训练与核心构件定位。补充 [Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)。

## TP 设计

[S4] [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)：理解 Transformer 内部如何切 attention heads 和 MLP，并在合适边界插通信。

[S5] [Megatron Core tensor_parallel.layers API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html)：ColumnParallelLinear、RowParallelLinear、VocabParallelEmbedding，以及复制参数去重的相关概念。对应源码目录 [tensor_parallel](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/tensor_parallel)。

[S6] [Megatron Core mappings API, 0.17.0](https://docs.nvidia.com/megatron-core/developer-guide/0.17.0/apidocs/core/core.tensor_parallel.mappings.html) 与 [mappings.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py)：重点读 copy/reduce 的 forward/backward 契约，不把所有 distributed collective 当成可直接替换的 autograd 运算。

本次读取 `mappings.py` 返回的文件 blob SHA 为 `846485ebae44fd5c26c8a16e3b72e6e343e5efe6`。它标识读取到的文件内容，不是整个仓库的 commit，也不是本项目代码的来源许可证声明。

## Pipeline、3D 并行与通信

[S7] [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)；[pipeline schedules API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.pipeline_parallel.schedules.html)；[schedules.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)。

先理解 DP/TP/PP 如何组合，再看 no-pipeline、non-interleaved 和 interleaved 的调度差别。本仓库 **不是** 对最新 schedules.py 的逐函数等价复刻。本次读取 schedules.py 的文件 blob SHA 为 `a703bc7c2f7a4ae4ccb0a47a51d36c92f4745106`。

[S8] [PyTorch 2.10 distributed 文档](https://docs.pytorch.org/docs/2.10/distributed.html)：Gloo/NCCL 后端、ProcessGroup、new_group 调用顺序、send/recv 的 global rank 与 NCCL tag 限制、分布式调试变量。本仓库实际验证使用 2.10.0 CPU 版。

[S9] [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)：本周 PP 选择 fill-drain 作为教学基线的阅读资料。我们的调度保留该种前后向组织方式，但没有复刻 activation rematerialization 或性能实现。

## 从自己的代码回看 Megatron

| 本仓库模块 | Megatron 阅读入口 | 本周只保留什么 |
|---|---|---|
| distributed/topology.py、mesh.py | core/parallel_state.py | 3D group 关系、rank 坐标和生命周期 |
| tensor_parallel/mappings.py | core/tensor_parallel/mappings.py | copy/reduce 两种 autograd 契约 |
| tensor_parallel/layers.py | core/tensor_parallel/layers.py | 三个基础 TP layer |
| tensor_parallel/loss.py | core/tensor_parallel/cross_entropy.py | 稳定的 vocab-parallel CE |
| model/parallel.py | core/transformer、core/models/gpt | 固定 tiny GPT，不做模型族适配 |
| pipeline_parallel/p2p.py | core/pipeline_parallel/p2p_communication.py | 固定 shape 的相邻 stage P2P |
| pipeline_parallel/schedules.py | core/pipeline_parallel/schedules.py | 调度职责边界；具体算法选择简化的 fill-drain |
| training/grad.py | core/distributed、core/optimizer | 梯度平均与逻辑 norm，不做 distributed optimizer |
| checkpoint.py | core/dist_checkpointing | 同拓扑 local shard 保存，不做可重分片格式 |

这里是概念和目录职责对应，不意味着接口兼容、性能相当或 checkpoint 互通。原仓库当前目录可从 [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) 继续浏览。
