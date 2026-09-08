# v3 (2026-09-08)

- Preserve T01–T60; add T61–T72, explicit zero3, CPU offload and SAC exercises.
- Separate pure SAC checkpointing from full-shard cache/replay; no silent nesting fallback.
- Add placement validation, transfer ownership helpers, op execution observer, local/multiprocess checks.
- Add CPUOffloadPolicy option to official FSDP2 reference (CUDA not tested here).
- Archive v2 docs; new validation distinguishes starter, temporary filled probes, and untested integration.

# 0.2.0

将v1之外的训练机制纳入必修，新增FSDP手写与官方对照路径。
原20个任务保留，新增T21–T60；baseline保持不变。
新增SP/CP/1F1B/VPP/MoE/ZeRO/FSDP/overlap/precision/model/text/checkpoint模块与实验框架。
新增明确EP-on-DP拓扑、GradRole语义、FSDP重算式生命周期和组合TrainRuntime。
新增数值oracle、typed P2P、官方FSDP2 CUDA reference、profile harness、七天密集日程。
明确区分starter通过、临时填充验证、尚待接线与GPU未测；旧版范围说明归档到docs/v1/。
