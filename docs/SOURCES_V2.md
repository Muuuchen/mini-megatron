# 官方资料与阅读映射

检索日期：2026-09-08。以下用于核验概念/API和完成练习后的源码阅读，不复制实现。
PyTorch对照代码按2.10文档编写；NVIDIA latest文档会演进，课程不是其全部功能的兼容保证。

| 本课 | 官方入口 |
|---|---|
| 总体parallelism | https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html |
| CP与SP区别 | https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html |
| Megatron FSDP | https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/megatron_fsdp.html |
| distributed optimizer | https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html |
| FSDP2参数生命周期/DTensor | https://docs.pytorch.org/docs/2.10/distributed.fsdp.fully_shard.html |
| AMP accumulation/unscale | https://docs.pytorch.org/docs/2.10/notes/amp_examples.html |
| activation checkpoint | https://docs.pytorch.org/docs/2.10/checkpoint.html |
| MoE dispatcher/EP/SP | https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html |
| communication overlap | https://docs.nvidia.com/nemo/megatron-bridge/latest/training/communication-overlap.html |
| Megatron源代码 | https://github.com/NVIDIA/Megatron-LM |

读完对应lab后再看Megatron的`tensor_parallel/mappings.py`、`pipeline_parallel/schedules.py`、
`transformer/moe/token_dispatcher.py`、`optimizer/distrib_optimizer.py`、`distributed/fsdp/`。
本课flat-FSDP+rematerialization只是教学算法，不是官方FSDP2的per-parameter DTensor内部实现。
