# 七天实践顺序 · v3

目标保留全部机制（不含FP8），新增内容依靠共用core和独立实验插入原有实践周。不是要求从零复刻FSDP2/DeepSpeed所有内部实现。一周是学习节奏，不保证全部组合与GPU性能都在这一周验收。

| 阶段 | 原课程主线 | v3新增重点 |
|---|---|---|
| Day 1 | T01–T07、T10–T13/T15、T21、T51–T54：模型/TP基础/拓扑 | 阅读三种内存策略及owner关系，不重复实现模型 |
| Day 2 | T08/T09/T14、T22–T25：词表CE、DP、SP | 确认grad mean/sum契约，为offload提供最终local grad |
| Day 3 | T26–T29/T33/T47/T48/T58：CP、重算、精度 | T65/T66：transfer/activation offload；T67/T68：SAC |
| Day 4 | T16/T17/T30–T32/T34/T56/T59：PP/1F1B/VPP/overlap | 对比copy event与P2P pending的生命周期，不重写调度 |
| Day 5 | T18/T35–T41：ZeRO1/2/FSDP、梯度范数 | T61/T62：ZeRO3；T63/T64：CPU state；T69/T70：SAC replay |
| Day 6 | T42–T46/T57：MoE/EP | 核对expert EDP与dense DP的CPU shard owner，不在错误group聚参 |
| Day 7 | T19/T20/T49/T50/T55/T60：组合、保存与恢复 | T71/T72：zero3_cpu_sac接线、snapshot与恢复 |

每次只运行已完成子集对应case，未完成算法不会fallback。原v2逐日命令在`docs/v2/WEEK_PLAN.md`作为参考；新任务命令如下。

## Day 3：SAC和传输先独立验

```bash
python -m pytest -q --exercises tests/test_v3_exercises.py -k 'sac and not zero3'
python -m megatron_mini.lab --case offload_transfer
python -m megatron_mini.lab --case offload_activation
```

CPU copy是快照/ownership模拟；并不发生HBM卸载。SAC必须同时检查参数梯度、dropout RNG和实际执行计数。

## Day 5：逐步组合参数分片

```bash
python -m megatron_mini.lab --case residency
torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case zero3
python -m megatron_mini.lab --case offload_optimizer
torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case offload_param
torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case zero3_sac
```

复用FlatLayout/AG/RS，不为ZeRO3和FSDP复制两份算法。offload路径CPU owner唯一，不以完整GPU参数副本撑起“假分片”。

## Day 7：独立检查与真正组合分开

```bash
python -m megatron_mini.lab --case offload_checkpoint
bash scripts/check_memory.sh
python -m megatron_mini.lab --recipe zero3_cpu_sac
# T55/T71以及recipe依赖完成后才能实际运行：
torchrun --standalone --nproc_per_node=2 \
  -m megatron_mini.lab --recipe zero3_cpu_sac --run --device cuda
```

最后命令不是现成训练后端，必须按T71补接线，并自己执行端到端dense/no-offload对照。要区分“模块数值通过”“组合通过”“真实GPU资源通过”三层。GPU trace/peak-memory未做就如实保留待验收，不用CPU测试代替。

进一步性能研究包括NVMe缓存、自动SAC策略/预算、异步pinned pool和跨全部并行维度组合；这些不增加第一周强制实现范围。主线CPU offload与SAC不是未来扩展。
