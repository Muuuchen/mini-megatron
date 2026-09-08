# v2 范围变更

旧版把1F1B、SP/CP、MoE/EP、ZeRO、重计算、混合精度和overlap放在主线之外。
本版这些机制及新增FSDP全部进入60项必修任务；入口是TASKS.md和WEEK_PLAN.md。
FP8仍不考虑。不以“生产级优化很多”为由要求把核心机制删掉。

简化的是实现范围：固定shape、dropless MoE、VPP=2、eager算子、明确的拓扑布局、flat checkpoint。
HF模型权重全兼容、任意拓扑弹性、fault-tolerant服务、全部硬件融合kernel不是“mini教学版”的完成定义。
历史v1说明在docs/v1/，不能作为v2完成度依据。
