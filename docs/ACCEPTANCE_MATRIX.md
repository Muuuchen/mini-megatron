# 由学生填写的验收矩阵

下列“未验收”指交付starter，无论目录/配置是否已经存在。临时填充副本仅见VALIDATION.md。
先通过数值，再测CUDA。至少记录3次更新和恢复后下一步；不以loss下降代替。

| 项目 | 最小配置 | forward/grad | 多次更新 | GPU memory/trace | 学生状态 |
|---|---|---|---|---|---|
| TP/SP | TP2 | 待填 | 待填 | 待填 | 未验收 |
| CP gather/ring | CP2 | 待填 | 待填 | 待填 | 未验收 |
| 1F1B/VPP | PP2,VPP2 | 待填 | 待填 | 待填 | 未验收 |
| ZeRO-1/2 | DP2 | 待填 | 待填 | 待填 | 未验收 |
| 手写FSDP | DP2,2+units | 待填 | 待填 | 待填 | 未验收 |
| 官方FSDP2 | CUDA DP2 | 待填 | 待填 | 待填 | 未验收 |
| MoE EP | DP2,EP2 | 待填 | 待填 | 待填 | 未验收 |
| BF16/FP16 | 单卡→双卡 | 待填 | 待填 | 待填 | 未验收 |
| 重算 | block/attention | 待填 | 待填 | 待填 | 未验收 |
| overlap | DP/TP/PP/EP/CP/FSDP | 时序待填 | 待填 | 待填 | 未验收 |
| dense_tp_zero2 | TP2 DP2 SP | 待填 | 待填 | 待填 | T55待接线 |
| pipeline_1f1b | PP2 | 待填 | 待填 | 待填 | T55待接线 |
| moe_ep_fsdp | DP4 EP2 EDP2 | 待填 | 待填 | 待填 | T55待接线 |
| checkpoint | 同拓扑 + flat2→1/4 | 待填 | 下一步待填 | 不适用 | 未验收 |

性能记录包含：设备型号/互联、PyTorch/CUDA/NCCL版本、global有效tokens、M、dtype、
max-rank step time、max-rank peak allocated memory、trace路径、串行baseline。
不要把DP副本、TP复制logits或CP子序列重复计入tokens/s。


## v3 memory矩阵（学生算法待补）

| case / recipe | 进程 | 检查内容 |
|---|---|---|
| residency | 1/2 | FULL状态、禁止重入、live-byte记账（非allocator证明） |
| zero3 | 2 | 两unit、M2、3步、dX/dW、clip、moments、AG/RS次数 |
| sac | 1/2 | 值/梯度/RNG、两个live cache、实际op执行次数 |
| zero3_sac | 2 | 参数分片同上，SAC replay不重复AG、不重跑saved addmm |
| offload_transfer | 1/2 | CPU快照 / CUDA H2D+D2H、host wait |
| offload_optimizer | 1/2 | CPUowner与moments、3步、overflow、resume |
| offload_param | 2 | 唯一CPUowner、staging释放、更新可见性 |
| offload_activation | 1/2 | saved tensor host副本、保护参数view、梯度 |
| offload_checkpoint | 1/2 | flush、隔离快照、恢复下一次更新 |
| zero3_cpu_sac recipe | 2 GPU | T55/T71待接线；上面独立case通过不代表这里通过 |

此表是验收设计，实际运行结果只以根目录VALIDATION.md为准。
