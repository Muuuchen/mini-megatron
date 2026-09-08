# v3 官方阅读入口与适用范围

核对日期：2026-09-08。本地API测试环境PyTorch 2.10.0+cpu；stable网页可能前移，以下PyTorch链接固定2.10。没有复制这些项目的生产实现。

- DeepSpeed ZeRO stages、offload配置和模型状态归属：
  https://deepspeed.readthedocs.io/en/latest/zero3.html
- DeepSpeed配置：parameter offload限ZeRO-3；optimizer offload覆盖ZeRO-1/2/3：
  https://www.deepspeed.ai/docs/config-json/
- PyTorch 2.10 FSDP2，尤其fully_shard与CPUOffloadPolicy：
  https://docs.pytorch.org/docs/2.10/distributed.fsdp.fully_shard.html
- PyTorch 2.10 SAC，CheckpointPolicy/create_selective_checkpoint_contexts、非reentrant和RNG：
  https://docs.pytorch.org/docs/2.10/checkpoint.html
- PyTorch 2.10 saved_tensors_hooks/save_on_cpu、pack/unpack契约：
  https://docs.pytorch.org/docs/2.10/autograd.html
- PyTorch作者对activation checkpoint与SAC取舍的解释：
  https://pytorch.org/blog/activation-checkpointing-techniques/
- NVIDIA Megatron Bridge模块级full/selective activation recomputation：
  https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html

本课的manual `_SelectiveUnit` 是教学组合设计，不是官方FSDP2 API或DeepSpeed源码。`observed_contexts`的TorchDispatch计数器用于验证实际执行，依赖PyTorch内部instrumentation接口，应固定版本回归。
