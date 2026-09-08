# megatron-mini · 一周分布式训练实验室

一个 **保留 20 个关键 TODO、用数值对照验收** 的 Megatron 风格教学仓库。
不是 NVIDIA 官方项目，不依赖 `megatron-core`，也不是删减后的 Megatron-LM fork。

目标不是“写一个 DDP 训练脚本”，而是亲手贯通：

```text
next-token batch
    -> tensor-parallel GPT stage
    -> microbatch / pipeline forward-backward
    -> data-parallel gradient averaging
    -> model-parallel-aware gradient clipping
    -> AdamW step
    -> same-topology checkpoint / resume
```

**交付状态：单进程 dense baseline、配置、数据、进程生命周期、参数切片初始化、P2P 传输、训练编排、日志和验收框架已经实现；20 个核心算法位置故意留空。**

`baseline` 可立即运行。`student` 只有补完相关 TODO 后才能运行；不会静默回退到参考实现。默认测试跳过学生练习，不能把默认测试通过理解成 TP/DP/PP 已实现。实际交付验证范围见 [VALIDATION.md](VALIDATION.md)。

## 1. 一周内具体做什么

| 一周主线 | 具体实现 | 刻意简化 |
|---|---|---|
| Dense GPT | MHA、pre-LayerNorm、GELU MLP、next-token CE | 学习式绝对位置编码；无 dropout、无 weight tying |
| TP | 通信 autograd、列/行并行 Linear、按 head 切 attention | 独立 Q/K/V；不做 fused QKV |
| 词表并行 | VocabParallelEmbedding、local LM head、稳定 CE | 词表均分，不支持 padding/ignore_index |
| DP | 不同样本、梯度累积、每步最终梯度求平均 | 不包装 DDP，不实现 bucket 和通信重叠 |
| PP | 连续层分段、相邻 stage P2P、GPipe-style fill-drain | 固定 shape/FP32；没有 1F1B 和 activation recompute |
| 正确性 | 比较 logits、loss、梯度、连续 AdamW 更新 | 小模型、可复现、FP32；不承诺 bitwise |
| 收尾 | 正确计算逻辑模型 grad norm、同拓扑断点恢复 | 每 rank 保存一份；不支持 reshard |

**不纳入一周必修：** MoE/EP、CP/SP、ZeRO/分布式优化器、FP16/BF16/FP8、Transformer Engine、fused kernels、异步通信重叠、virtual PP、弹性恢复、HF 权重兼容。扩展接口与验收要求写在 [EXTENSIONS.md](docs/EXTENSIONS.md)，不是已实现的功能。

模型刻意选用很小的 GPT，而不是先接 Qwen/Llama：这周的难点是训练语义和并行组合，不是模型适配。合成序列只是可学习、可复现的测试任务，loss 下降不代表获得了自然语言能力。

## 2. 立即开始

从解压后的仓库根目录执行。先使用已有的、适配自己设备的 PyTorch 环境；不要为了此实验盲目替换 GPU 环境里的 CUDA/PyTorch。

```bash
cd megatron-mini
python -m pip install -e '.[dev]'

# 不依赖 GPU、模型下载或任何学生实现。
python -m pytest -q
python -m megatron_mini.train --mode baseline --device cpu --steps 20

# 显示 20 项任务、源码位置和推荐日期。
python -m megatron_mini.tasks
```

也可 `bash scripts/day1.sh`，会把基准日志写到 `runs/baseline_cpu.jsonl`。

Python 要求为 3.10+，依赖声明为 PyTorch 2.5–2.x；**实际验证环境是 Python 3.13.5、PyTorch 2.10.0+cpu、pytest 9.0.2**，声明范围不等于每个版本都已测试。

Linux 是推荐实验环境。CPU 使用 Gloo，CUDA 使用 NCCL；当前版本不提供 MPS 后端。没有多张 GPU 也可以用多个 CPU 进程完成通信和数值练习，不能用 CPU 测量结果推断 GPU 吞吐。

## 3. 怎么做练习

每个实现点都有唯一编号，例如 `T06`。阅读函数 docstring，它包含输入/输出布局、需要保持的通信或梯度语义、常见错误。然后用自己的实现替换 `todo("T06")`。

```python
class RowParallelLinear(nn.Module):
    def forward(self, local_x):
        """[B,S,Hin/TP] -> [B,S,Hout]。
        本地无 bias matmul -> TP SUM -> 最后加 replicated bias。
        输入已分片，不再 split；不能把 bias 累加 TP 次。
        """
        todo("T06")
```

`todo()` 抛出带编号的异常。入口在启动通信前检查所需编号，防止运行到一半才撞上占位符。移除占位符后状态只是“待验收”，**不是自动通过**。

```bash
# 第一天只验收 loss；不要一开始执行所有未完成练习。
python -m pytest -q --exercises tests/test_exercises.py -k token_loss

# 补完所有单进程练习后，才运行全部 exercise tests。
python -m pytest -q --exercises
```

完整任务说明见 [TASKS.md](docs/TASKS.md)；逐日的精确命令见 [WEEK_PLAN.md](docs/WEEK_PLAN.md)。

## 4. 仓库结构

```text
megatron-mini/
├── README.md / VALIDATION.md
├── pyproject.toml / Makefile
├── configs/
│   ├── tiny.json                 # 正确性优先
│   └── small_gpu.json            # 补完后自行做 GPU 实验
├── megatron_mini/
│   ├── train.py                  # baseline/student CLI，已完成
│   ├── config.py / data.py        # 固定形状、step-indexed 数据，已完成
│   ├── tasks.py / exercise.py     # TODO 进度与启动检查，已完成
│   ├── distributed/
│   │   ├── topology.py           # T02: group plan；坐标运算已给出
│   │   └── mesh.py               # ProcessGroup 生命周期和日志规约
│   ├── model/
│   │   ├── dense.py              # 可运行的 dense oracle
│   │   ├── parallel.py           # T10–T13: TP GPT 与本地 PP stage
│   │   └── state.py              # 相同 dense 参数切片初始化
│   ├── tensor_parallel/
│   │   ├── mappings.py           # T03–T04: 自定义 autograd
│   │   ├── layers.py             # T05–T07: column/row/embedding
│   │   └── loss.py               # T08–T09: vocab-parallel CE
│   ├── pipeline_parallel/
│   │   ├── p2p.py                # 固定形状 send/recv，已完成
│   │   └── schedules.py          # T16–T17: GPipe 前后向与图生命周期
│   ├── training/
│   │   ├── loss.py               # T01: 普通 CE
│   │   ├── schedules.py          # T15: 无 PP 的 microbatch 累积
│   │   ├── grad.py               # T14/T18: DP average / logical norm
│   │   └── engine.py             # 编排 zero_grad -> schedule -> sync -> step
│   └── checkpoint.py             # T19–T20: 保存/恢复
├── tests/
│   ├── test_ready.py             # 已完成脚手架的测试
│   └── test_exercises.py         # 显式启用的学生验收
├── tools/check_distributed.py    # 真正多进程的数值对照驱动
├── scripts/day1.sh / check_cpu.sh
└── docs/
    ├── WEEK_PLAN.md / TASKS.md
    ├── ARCHITECTURE.md
    ├── DEBUGGING.md
    ├── SOURCES.md                # Megatron 原仓库对应模块/阅读入口
    ├── EXTENSIONS.md
    └── EXPERIMENTS.md            # 自行填写的测量记录
```

## 5. 一周路线

按已有 PyTorch/autograd 基础、每天约 3–5 小时的学习预算安排，总体约 25–35 小时；这是工作量规划，不是完成保证。第二天和第五天预留较多调试空间。

| 日期 | 要补的编号 | 当天交付物 |
|---|---|---|
| Day 1 | T01 | dense baseline 跑通；理解 batch、loss、step；CE 单测通过 |
| Day 2 | T02–T07 | 进程组、通信 autograd、并行 Linear/Embedding 的双进程对齐 |
| Day 3 | T10–T13、T15 | student 的 TP=PP=DP=1 模型和梯度累积对齐 |
| Day 4 | T08–T09、T14 | TP=2、DP=2 的真实训练和词表 CE 对齐 |
| Day 5 | T16–T17 | PP=2 fill-drain 跑通；验证 M=1 和 M=3 |
| Day 6 | T18–T20 | 去重梯度裁剪、optimizer/RNG/data cursor 恢复 |
| Day 7 | 集成与测量 | 混合并行矩阵、3 次连续更新对齐、实验记录 |

## 6. 补完后如何启动训练

下面命令**不是开箱即用的已完成分布式模型**，运行前会检查对应 TODO。

```bash
# 单进程 student。
python -m megatron_mini.train --mode student --steps 20

# 两个 CPU 进程运行 TP、DP 或 PP。三个命令分别启动，互不依赖。
torchrun --standalone --nproc_per_node=2 -m megatron_mini.train --mode student --tp 2
torchrun --standalone --nproc_per_node=2 -m megatron_mini.train --mode student --dp 2
torchrun --standalone --nproc_per_node=2 -m megatron_mini.train --mode student --pp 2

# 四进程 TP×PP；WORLD_SIZE 必须恰好等于 TP*PP*DP。
torchrun --standalone --nproc_per_node=4 -m megatron_mini.train --mode student --tp 2 --pp 2
```

同一套入口加 `--device cuda` 切换 CUDA/NCCL，必须一进程一张可见 GPU；不要把两个 NCCL rank 塞进一张卡冒充双卡训练。GPU 路径留待你的实际硬件验收。

```bash
# 有两张可见 GPU 且对应 TODO 完成时。
torchrun --standalone --nproc_per_node=2 -m megatron_mini.train   --mode student --device cuda --tp 2 --config configs/small_gpu.json   --micro-batch 2 --microbatches 4 --steps 20 --output runs/tp2_gpu.jsonl
```

## 7. 验收标准不是 loss 能下降

单算子检查前向值、输入梯度、权重梯度；端到端对照同 seed、同逻辑 global batch 的 dense GPT，默认检查 **连续 3 次 AdamW 更新**。最后 stage 才计算 loss，所有 stage 的本地参数都与对应 dense shard 对照。

```bash
# 这些命令都要求相应练习已完成。
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case linear --tp 2
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case loss --tp 2
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case step --pp 2 --clip 0.25

# 完整 CPU 矩阵；默认最多 2 进程。
bash scripts/check_cpu.sh
# 主机内存足够时，扩展到所有两维组合及真正 3D 的八进程测试。
MAX_PROCS=4 bash scripts/check_cpu.sh
MAX_PROCS=8 bash scripts/check_cpu.sh
```

最容易漏掉的错误包括：Column backward 没合并 dX；Row 的 bias 被重复相加；CE 梯度多除 TP；loss 已除 M 后上游 stage 又除 M；DP 梯度在错误 group 上平均；复制参数在 global norm 中重复计数。定位顺序见 [DEBUGGING.md](docs/DEBUGGING.md)。

## 8. 必须明确的实现边界

FP32 是主线；PP 传输硬编码 FP32，不能只改参数 dtype 就宣称支持 BF16。batch/seq_len/microbatch 大小固定相等，无 padding、无变长、无 dropped tokens，所以 `loss/M` 与 DP 平均才对应全局 token mean。真实不等长任务需重新设计 loss denominator。

PP 用的是 **GPipe-style fill-drain 调度，不是 Megatron 的 1F1B，也没有 GPipe 论文中的 activation rematerialization**。每个 microbatch 保存图直到反向；stage 只持有本地层，但激活开销可随 M 增长。

每个 rank 临时在 CPU 构建 tiny dense 参数做初始化；每个 TP/PP rank 用相同 DP 坐标独立生成合成 batch。这是为数值可比做的教学捷径，不是生产级权重加载或数据加载设计。

checkpoint 只保证单文件原子替换，不保证全体 rank 的事务提交。CLI 限制恢复时模型、TP/PP/DP、rank、设备类型、microbatch 设置、学习率、裁剪设置一致；`--steps` 是累计总步数，可以增大。数据是 step-indexed，没有真实 DataLoader worker 状态。

## 9. 推荐阅读顺序

先读本 README，再按当天任务读 `WEEK_PLAN.md` 和 `TASKS.md`；遇到接口或梯度疑问读 `ARCHITECTURE.md`；对齐失败读 `DEBUGGING.md`。完成一个子模块后，通过 `SOURCES.md` 回到 Megatron 对应实现，而不是从原仓库入口一路追所有生产依赖。
