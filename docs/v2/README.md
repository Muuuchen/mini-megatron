# megatron-mini v2 · 全机制七天实验课程

**不考虑 FP8；TP、SP、CP、PP/1F1B/VPP、MoE/EP、ZeRO、FSDP、激活重计算、BF16/FP16 和通信重叠全部纳入课程。**
这是原创的教学 scaffold，不是 NVIDIA 官方项目，也不复制 Megatron 的生产实现。沿用 v1 的可运行 dense baseline，新增机制按独立实验拆开，最终接入统一训练生命周期。

**交付是带 60 个核心 TODO 的课程仓库，不是已经实现上述全部功能的框架。**
20 个原任务保留，新增 40 个；每个任务有中文代码契约、依赖和验收入口。`todo()` 明确失败，不回退到 dense/FSDP2 冒充手写完成。`train.py` 保留 v1 的 FP32/GPipe 入口；新机制走 `lab.py`，组合训练的接线留在 **T55**。

## 1. 七天到底完成什么

本周目标是**每种机制有一个最小正确版本，以及几个关键组合通过数值验收**。不把机制移回“以后再说”，也不要求第一周获得所有配置组合的生产性能。

| 机制 | 必修实现 | 控制难度的取舍 |
|---|---|---|
| TP + SP | Column/Row Linear、词表 CE、SP AG/RS、自定义 backward、复制参数梯度 | 不写 fused CUDA kernel；SP 先验 MLP+LN 小图，再接 GPT |
| CP | KV gather 基准、全局 causal mask、ring online softmax、dK/dV 回 owner | 固定长度、连续切序列；ring 不做负载均衡花式切块 |
| PP | GPipe、非交错 1F1B、VPP=2 interleaving、typed P2P、图生命周期 | 固定 activation shape；VPP 可先要求 M 是 PP 的倍数 |
| MoE/EP | top-2 router、aux loss、permutation、变长 AllToAll、unpermutation、EDP | 4 个专家、dropless、普通 PyTorch 专家 MLP |
| ZeRO | ZeRO-1 optimizer shard、ZeRO-2 grad RS、parameter AG | 共用 FlatLayout；完整 buffer 是数值基准，内存验收需 bucket |
| FSDP / ZeRO-3 | 参数 shard→AG→计算→释放→反向 AG→grad RS→local optimizer | 手写版用**单元重算**简化 saved tensors；不是伪造 FSDP2 hooks |
| 官方 FSDP 对照 | FSDP2 bottom-up `fully_shard`、MP policy、optimizer 构造顺序 | 独立 CUDA reference；不能替代手写任务 |
| 精度与重算 | FP32 master、BF16/FP16 autocast、全局 overflow、block/attention recompute | 复用 PyTorch kernels/checkpoint；FP8 排除 |
| 通信重叠 | DP bucket、FSDP prefetch、TP chunks、PP异步P2P、EP/CP双buffer | 先验顺序与buffer生命周期，再用CUDA trace证明重叠 |
| 模型与文本 | RMSNorm、RoPE、GQA、SwiGLU结构、本地UTF-8字节样本 | 不先接HF权重或下载真实大模型 |
| 收尾 | role-aware梯度、去重norm、scaler/RNG/checkpoint、flat reshard | reshard先做离线2→1/4，不宣称任意TP/EP在线弹性 |

原先每天 3–5 小时、25–35 小时的估计不再适合这个范围。本版按**已有 PyTorch/autograd/分布式基础的一周密集实践**设计，建议预留约 45–60 小时作为课程工作量预算；这不是完成保证。若只投入原来的时数，就不应承诺全部正确性和 GPU 性能验收也同时完成。

## 2. 开箱运行与开始做题

使用已有的 PyTorch 环境。核心依赖仍为 `torch>=2.5,<3`；这只是声明范围，实际验证是 **Python 3.13.5 / PyTorch 2.10.0+cpu**。官方 FSDP2 对照按 PyTorch 2.10 API 编写，CUDA 未在本次环境测试。

```bash
cd megatron-mini-v2
python -m pip install -e '.[dev]'
python -m pytest -q
python -m megatron_mini.train --mode baseline --device cpu --steps 3
python -m megatron_mini.tasks
python -m megatron_mini.lab --list
```

离线且已有setuptools/torch/pytest时，可用 `python -m pip install -e . --no-deps --no-build-isolation`。

默认测试的 skip 是未启用学生任务，**不是算法验证通过**。移除 TODO 仅代表进入“待验收”，不是已完成。

独立实验可以不先补完 20 个 v1 任务。例如学习手写 FSDP，只需先实现 T35、T38、T39：

```bash
# 现在直接执行会在创建进程组前指出缺失 TODO。
torchrun --standalone --nproc_per_node=2 \
  -m megatron_mini.lab --case fsdp

# 完成 SP 的四个任务后，单独对齐前向和梯度。
torchrun --standalone --nproc_per_node=2 \
  -m megatron_mini.lab --case sp

# 此命令只展示配置与依赖，不声称集成已实现。
python -m megatron_mini.lab --recipe moe_ep_fsdp
```

组合训练必须再完成 T55 和该 recipe 的依赖：

```bash
torchrun --standalone --nproc_per_node=4 \
  -m megatron_mini.lab --recipe dense_tp_zero2 --run
```

## 3. 仓库结构与两种入口

```text
megatron_mini/
├── train.py / model/dense.py       # 原baseline与TP/DP/GPipe练习入口，保留
├── lab.py                         # 新增：机制实验、recipe计划、集成启动
├── distributed/
│   ├── axis.py                    # 已完成：显式SUM/AG/RS适配
│   └── topology_v2.py             # TP/PP/CP/DP + EP/EDP 子组
├── sequence_parallel/             # AG/RS autograd、SP MLP/LN
├── context_parallel/              # KV gather baseline、online/ring
├── pipeline_parallel/
│   ├── planner.py                 # 1F1B/VPP事件、已完成的依赖模拟器
│   ├── executor.py                # 不绑定具体模型的前后向executor
│   └── transport.py               # 已完成typed send/recv；IO接线TODO
├── moe/                           # router、dispatcher、专家前向框架
├── optimizer/                     # FlatLayout、ZeRO-1/2
├── fsdp/                          # 手写单元生命周期 + 官方FSDP2对照
├── runtime/                       # dtype/scaler、overlap、grad roles、ckpt、text
├── training/engine_v2.py           # step顺序已给；build_training_runtime是T55
├── model/llama.py                 # RMSNorm/RoPE/GQA任务、结构已接好
└── labs/                          # manifest、case/recipe目录、独立数值oracle

tests/test_v2_ready.py              # 已完成脚手架测试
tests/test_v2_exercises.py          # 学生算法测试，显式 --exercises
examples/torch_fsdp2_reference.py   # 官方FSDP2 CUDA对照，不是学生答案
docs/                              # 日程、任务、架构、FSDP专题、接线验收
validation_logs/                   # 本次实际测试日志，区分starter与临时填充
```

## 4. 三个不要混淆的关系

**FSDP 不是在 TP 上换个通信函数。** TP 的 GEMM 消费分片权重；FSDP 的每个副本消费不同数据，在执行某个 unit 前临时恢复该 unit 的完整参数。FSDP 的“完整”是相对当前 TP/PP 局部模块而言，不是 gather 整个训练模型。

**ZeRO-3 与 FSDP 不需要重复写两套相同的分片系统。** 本课程用 `sharding=ddp/zero1/zero2/fsdp/torch_fsdp2` 选择策略，ZeRO-3 粒度的参数/梯度/optimizer sharding 在手写 FSDP 中实现；官方 FSDP2 是对照，不是另一个相乘的并行维度。

**SP 不额外乘卡数；EP 也不能在本教学拓扑上重复相乘。** 本版 `WORLD=TP×PP×CP×denseDP`，EP 是 denseDP 内的子组，`EDP=denseDP/EP`。例如 TP=2、DP=4、EP=2 用 8 个进程，不是 16 个。FSDP 的分片组对 dense 参数和 expert 参数不同。完整参数归属见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 5. 完成判据与验证状态

每个模块至少验证 forward、input/parameter gradients；ZeRO/FSDP case 比较连续 3 次 AdamW 更新及 moments。最终集成至少验 dense TP×ZeRO-2、PP 1F1B、MoE EP×FSDP；四维/五维任意组合不是自动支持。

交付 starter 当前：**37 passed，31 skipped；60 个 TODO 全部保留。** 已实测 dense baseline 和 CPU 双进程通信适配。临时填充的副本通过了 SP、CP gather、ZeRO-1/2、FSDP、MoE EP、1F1B 的双进程相应数值测试，答案不随仓库交付。**Ring backward、VPP transport、真实 CUDA overlap、T55 集成均未完成验收。** 详见 [VALIDATION.md](VALIDATION.md)。

完整课程入口：[七天计划](docs/WEEK_PLAN.md) → [60 项契约](docs/TASKS.md) → [FSDP 专题](docs/FSDP_AND_ZERO.md) → [集成接线](docs/INTEGRATION.md)。旧版文档只在 `docs/v1/` 作为历史记录，不再决定本周范围。
