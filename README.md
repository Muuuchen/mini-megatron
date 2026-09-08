# megatron-mini v3 · ZeRO-3 / CPU Offload / SAC

**除 FP8 外，保留 v2 的 TP/SP/CP/PP/1F1B/VPP/MoE/EP/ZeRO/FSDP、BF16/FP16、重计算与 overlap；新增显式 ZeRO-3、CPU offload 和 SAC。**
这是可运行 baseline + 待补核心算法的教学仓库，不是功能已全部实现的训练框架。

**72 个核心 TODO：原 T01–T60 保留，新增 T61–T72。** 不把“新增配置成功”当作算法完成。新机制有独立数值实验，组合训练仍需完成 T55/T71。

## 本次新增什么

| 机制 | 教学范围 | 关键不变量 |
|---|---|---|
| 显式 ZeRO-3 | `sharding='zero3'`、逐 unit 参数/梯度/optimizer state 分片 | 计算外只持久持有本地 shard；不在 step 后 AllGather 全模型 |
| 手写 FSDP + ZeRO-3 共用内核 | 共用 FlatLayout、参数 AG、梯度 RS | 两个课程入口，不把官方 FSDP2 与 DeepSpeed 实现说成完全相同 |
| CPU optimizer offload | CPU FP32 master、Adam m/v、最终 grad shard、CPU step | 不重复 DP 平均；overflow 时所有 owner 不更新 |
| CPU parameter offload | CPU canonical shard → H2D → unit AG → compute → release | CPU owner 唯一；NCCL 不接收 CPU tensors |
| Activation offload | saved_tensors_hooks 的 pack/unpack，保护参数 storage | 真正复制到 host；不能在 pack 对象里长期保留 GPU activation |
| SAC | 按 ATen op 保存昂贵结果、重算便宜算子 | fresh cache per invocation；检查执行次数，不只是 policy 命中次数 |
| full-shard + SAC | 独立 cache-forward / replay-backward 任务 | collective 在 SAC 外；不能只在整单元重算外套 checkpoint |
| 收尾 | CPU owner snapshot、恢复下一步、组合训练接线 | 保存前 drain；checkpoint 不保存 transient full buffers/SAC caches |

Offload 主线是 **GPU↔CPU**，包括参数、梯度/optimizer 和激活三个对象类别。NVMe 的存储接口与验收边界写在专题文档中，但本版不提供 NVMe 执行路径，也不冒称 ZeRO-Infinity。SAC 指 **Selective Activation Checkpointing**；Megatron 的模块级 selective recompute 同时保留为 T33。

## 立即运行

```bash
cd megatron-mini-v3
python -m pip install -e '.[dev]'
python -m pytest -q
python -m megatron_mini.train --mode baseline --device cpu --steps 3
python -m megatron_mini.tasks
python -m megatron_mini.lab --list
```

无网络且依赖已安装时：`python -m pip install -e . --no-deps --no-build-isolation`。
默认测试只验收已实现脚手架；`--exercises` 才会执行留空算法，未补完时失败是预期行为。

### 只学新增机制，不必先做完 60 个旧题

```bash
# 查看配置与依赖，不运行训练、不创建通信组。
python -m megatron_mini.lab --recipe zero3_cpu_sac

# 补齐相应 TODO 后运行；未完成会在创建 communicator 前明确退出。
python -m pytest -q --exercises tests/test_v3_exercises.py -k sac
python -m megatron_mini.lab --case offload_optimizer

torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case zero3
torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case zero3_sac
torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case offload_param
```

独立 case 的具体依赖在 [新增任务契约](docs/MEMORY_TASKS.md)。例如 SAC 只需 T67/T68；CPU optimizer 只需 T63/T65。

```bash
# 所有新增独立 case；CPU 的 copy 只验证快照/生命周期，不是实际卸载。
bash scripts/check_memory.sh
# 真正两张可见 GPU，完成相关 TODO 后再做：
DEVICE=cuda PROCS=2 bash scripts/check_memory.sh

# 完成 T55/T71 和全部 recipe 依赖后才可训练；不是开箱即用。
torchrun --standalone --nproc_per_node=2 \
  -m megatron_mini.lab --recipe zero3_cpu_sac --run --device cuda
```

## 新增目录

```text
megatron_mini/
├── memory/
│   ├── residency.py         # T61: full buffer 生命周期；计数器不是显存证明
│   └── integration.py       # T71: memory recipe → TrainRuntime
├── optimizer/zero3.py       # T62: 明确的ZeRO-3逐unit入口，复用FSDP核心
├── offload/
│   ├── config.py            # 已给：合法组合和拒绝静默降级
│   ├── transfer.py          # T65: copy+event；wait_host/wait_stream已给
│   ├── optimizer.py         # T63: CPU Adam shard，状态保存/恢复已给
│   ├── parameters.py        # T64: CPU owner → GPU staging → full unit
│   ├── activations.py       # T66: pack；unpack/drain/hooks已给
│   └── checkpoint.py        # T72: flush和CPU-only snapshot
├── runtime/sac.py           # T67/T68: policy+checkpoint；执行计数器已给
├── fsdp/selective.py        # T69/T70: 分片参数下的选择性replay
└── labs/memory_checks.py    # 已给：值、梯度、更新、moments、缓存行为对照

tests/test_v3_ready.py       # 已实现脚手架
tests/test_v3_exercises.py   # 未补算法的验收
scripts/check_memory.sh
```

v2 的全部代码和任务保留。旧 README、旧验收记录在 `docs/v2/`；**它们是历史记录，不是 v3 的新测试结果**。

## 正确性和范围

ZeRO-3/FSDP 分片模型状态；offload 决定存放位置；SAC 决定哪些激活保存/重算。这是三个可组合维度，不能用同一个布尔值表示。

`zero3` 和 `fsdp` 是本 mini 的两个入口，共享手写 full-shard 生命周期。不是先套 FSDP 再套 ZeRO-3，更不是多乘一维卡数。原拓扑仍为 `WORLD = TP × PP × CP × denseDP`，EP 是 denseDP 的子组，专家只能在对应 EDP owner group 归约/分片。

本课参数尺寸小、固定 shape、无 tied weights、无 mutable buffers；先 FP32 对齐，再验 BF16/FP16，FP8 不纳入。CPU/Gloo RS fallback 是 SUM+slice，不能用于证明 NCCL 通信量或 GPU 内存收益。

**SAC 与本手写 FSDP 的组合不能依靠双层 checkpoint 自动获得。** T69/T70 让 paired SAC cache 跨原始 forward 和反向 replay 存活，且不长期保留完整参数。SAC 保存的 op 输出仍占内存；generic activation hooks 不会自动把 SAC 的 Python cache 卸载到 CPU。完整限制见 [专题](docs/ZERO3_OFFLOAD_SAC.md)。

## 一周怎么安排

继续按七个实践阶段组织，新增工作集中到 Day 3、5、7。通过共用分片/传输/重算工具控制重复工作，而不是删掉机制。一周的目标是小规模机制和指定组合；**不是所有配置笛卡尔积、故障恢复和 GPU 性能都全面验收**。原 45–60 小时只是 v2 的课程预算，不是新增功能后的完成保证。

阅读顺序：[七天计划](docs/WEEK_PLAN.md) → [新增12项契约](docs/MEMORY_TASKS.md) → [内存策略专题](docs/ZERO3_OFFLOAD_SAC.md) → [全部任务](docs/TASKS.md) → [当前实测记录](VALIDATION.md)。

## 当前交付验证

原样starter **50 passed / 40 skipped，72个TODO保留**；临时填充副本的9个新增case通过单进程和真实双进程CPU验证。完整memory训练recipe、真实GPU offload/SAC性能未验证。依赖声明torch>=2.5不等于逐版本兼容；实际环境是2.10.0+cpu。详见[VALIDATION.md](VALIDATION.md)。
