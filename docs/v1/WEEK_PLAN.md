# 七天实现顺序与验收命令

所有命令在仓库根目录运行。CPU 命令启动真实进程与 Gloo 通信；不需要 GPU。预计每天约 3–5 小时，若不熟悉 autograd.Function 或 torchrun，要额外预留基础学习时间。

完成一个编号后，运行最小相关测试；不要等最后一天才启动多进程。每天建议提交一次本地 Git commit，提交信息可用 `day2: TP layers and gradient equivalence` 这种形式。

## Day 1：把一次训练 step 读透

先读 `model/dense.py -> data.py -> train.py`，运行 baseline，找到 logits/targets 的形状、loss reduction、optimizer 生命周期。实现 T01，不重写已经给出的 dense Transformer。

```bash
python -m pytest -q
python -m megatron_mini.train --mode baseline --steps 20 --output runs/day1.jsonl
python -m pytest -q --exercises tests/test_exercises.py -k token_loss
```

验收：能够解释 data 已 shift labels，为何 loss 里不能再 shift；知道 microbatch、local batch、global batch 的关系；CE forward 和 gradient 通过对照。

## Day 2：进程组与 TP 基础层

按 T02 -> T03/T04 -> T05/T06 -> T07 实现。先验证纯 rank arithmetic，再验证两进程。阅读 `distributed/mesh.py` 已给出的 group 生命周期，不要改成多个零散全局变量。

```bash
python -m pytest -q --exercises tests/test_exercises.py -k group_plan
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case groups --tp 2
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case linear --tp 2
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case embedding --tp 2
```

验收：column 的完整输入梯度、row 的本地输入梯度、非零 bias、跨 shard 的 embedding token 都正确。不能只对齐 forward。

## Day 3：组合成完整 student GPT

实现 T10/T11/T12/T13，再做 T15。先在 TP=PP=DP=1 下组合，避免同时排查模型结构和词表通信。

```bash
python -m pytest -q --exercises tests/test_exercises.py -k 'student_model or student_gradient'
python -m megatron_mini.train --mode student --steps 20 --output runs/day3.jsonl
```

验收：student 与 dense logits/每个参数梯度一致；多 microbatch 累积等价于同一大 batch；每步只更新一次。此时不要求 vocab-parallel CE 完成，所以先不要运行 TP=2 的整模型训练。

## Day 4：词表并行和 DP

实现 T08/T09 的稳定 CE，再实现 T14。DP 必须使用不同数据，但同一 TP/PP 副本使用相同数据。多进程测试中 --case step 默认比较连续 3 次 AdamW 更新。

```bash
python -m pytest -q --exercises tests/test_exercises.py -k single_rank_vocab_loss
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case loss --tp 2
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case model --tp 2
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case step --tp 2
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case step --dp 2
```

验收：大幅值 logits 不溢出，target 位于不同 shard 时都正确，global mean loss 与 dense 一致。机器资源允许时加 `--nproc_per_node=4 --tp 2 --dp 2` 的 step 测试。

## Day 5：GPipe-style 流水线

读 `p2p.py`，按 T16/T17 实现缓存图和跨进程反向。不要从 1F1B 开始；当天重点是 forward/backward 的边界和梯度拥有者。

```bash
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case step --pp 2 --microbatches 1
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case step --pp 2 --microbatches 3
# 可选四进程，验证 TP lane 与 PP stage 的组合。
torchrun --standalone --nproc_per_node=4 -m tools.check_distributed --case step --tp 2 --pp 2
```

验收：首 stage embedding 与末 stage LM head 参数都能更新；中间 activation.grad 存在；loss 只缩放一次；每个 stage 只拥有其层；M=1 不挂起。

## Day 6：训练收尾语义

实现 T18/T19/T20。global grad norm 必须对 TP 复制参数去重；恢复不能只恢复 model weights。

```bash
python -m pytest -q --exercises tests/test_exercises.py -k 'clip_norm or checkpoint'
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case step --tp 2 --clip 0.25
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case step --pp 2 --clip 0.25
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case checkpoint --tp 2
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case checkpoint --pp 2
```

恢复的 CLI 示例（同一拓扑和 batch 设置）：

```bash
torchrun --standalone --nproc_per_node=2 -m megatron_mini.train   --mode student --tp 2 --steps 3 --checkpoint-dir checkpoints/tp2

torchrun --standalone --nproc_per_node=2 -m megatron_mini.train   --mode student --tp 2 --steps 6 --resume checkpoints/tp2
```

验收：resume 下一步 loss、参数和 optimizer moments 与连续训练一致；RNG 可重复；data cursor 没有跳步/重放；错误配置明确拒绝。

## Day 7：集成、解释与测量

```bash
python -m megatron_mini.tasks
bash scripts/check_cpu.sh
# 资源允许再执行：
MAX_PROCS=4 bash scripts/check_cpu.sh
MAX_PROCS=8 bash scripts/check_cpu.sh
```

两进程分别覆盖 TP、DP、PP；四进程覆盖两两组合；八进程才能同时让 TP/PP/DP 都大于 1。这些可以是 CPU 进程，不需要八张 GPU。较少进程的测试不能被描述成真正 2×2×2 联合验收。

有多 GPU 资源时，将对应 case 的 `--device` 改为 `cuda`，先跑 tiny 数值检查，再用 small_gpu 做初步性能测量。填写 [EXPERIMENTS.md](EXPERIMENTS.md)，记录测试环境、global batch、M、平均 step time、global tokens/s、max-rank peak allocated memory。

最终交付应是：20 个主线编号已通过对应测试；至少 TP=2/DP=2/PP=2 的数值对齐；一份诚实标明已测与未测环境的实验记录；能够解释 TP mapping、DP loss scale、PP 图边界这三个关键问题。1F1B/SP/overlap 不计入本周完成度。
