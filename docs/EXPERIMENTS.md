# 实验记录模板（请自己测量，不把占位项当结果）

## 环境与工作负载

记录日期、CPU/GPU 型号与数量、显存、互联、driver、PyTorch/CUDA/NCCL 版本、代码 commit；记录模型 vocab/seq/hidden/heads/layers/ffn、FP32、micro_batch、M、global batch、TP/PP/DP、optimizer 和 clip 设置。

本周的数值验收目标是功能正确，不是追平 Megatron 的吞吐。默认 tiny config 太小，适合调试，不适合外推到大模型训练。

## 数值验收

| 配置 | loss | gradients | 3-step AdamW parameters | clipped grad norm | resume | 环境 |
|---|---|---|---|---|---|---|
| TP1 PP1 DP1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待填 |
| TP2 PP1 DP1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待填 |
| TP1 PP2 DP1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待填 |
| TP1 PP1 DP2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待填 |
| TP2 PP2 DP1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待填 |
| TP2 PP1 DP2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待填 |
| TP1 PP2 DP2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待填 |
| TP2 PP2 DP2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待填 |

具体某一项的“通过”只表示该形状/种子/实现/后端下测试通过，不是所有模型与环境的保证。max_abs_error 混合了各类被测张量，不能简单称作 loss error。

## 性能测量

训练 CLI 输出 step_ms、global_tokens_per_second、max_rank_peak_allocated_bytes，前两步标为 warmup 并排除出最终均值。计时前同步，结束后等待 CUDA，取 ranks 中最大 step elapsed。峰值内存是 PyTorch allocated memory 的最大 rank 值，不等于 nvidia-smi 进程总占用，也不是所有卡内存之和。

有效吞吐计算：`micro_batch * M * DP * seq_len / step_seconds`，不能再乘 TP 或 PP。输入数据生成不计入 step interval；因此这不是包含存储与 tokenizer 的端到端吞吐。日志规约在 student step 内，性能解释时应包含这项开销。

baseline 和 PP=1 student 都以 microbatch 逐个前后向累积。比较不同 DP 度时，固定 global batch：例如单进程 micro_batch=2/M=4，对比 DP2 micro_batch=2/M=2。这样虽然 global batch 相同，M 不同仍影响调度，应明确记录。另做固定每 rank workload 的 weak-scaling 实验时，要写清 global batch 已改变，不能混作相同工作负载的提速。

```bash
# 已完成 student、具备两张 GPU 后，用同 global batch 做初步比较。
python -m megatron_mini.train --mode baseline --device cuda   --config configs/small_gpu.json --micro-batch 2 --microbatches 4   --steps 20 --output runs/baseline_gpu.jsonl

torchrun --standalone --nproc_per_node=2 -m megatron_mini.train   --mode student --device cuda --tp 2 --config configs/small_gpu.json   --micro-batch 2 --microbatches 4 --steps 20 --output runs/tp2_gpu.jsonl
```

| 实验 | global batch | TP/PP/DP | M | mean steady-state step ms | global tokens/s | max-rank peak allocated MiB |
|---|---|---|---|---|---|---|
| dense baseline | 待填 | 1/1/1 | 待填 | 未测 | 未测 | 未测 |
| TP | 待填 | 2/1/1 | 待填 | 未测 | 未测 | 未测 |
| DP | 待填 | 1/1/2 | 待填 | 未测 | 未测 | 未测 |
| PP fill-drain | 待填 | 1/2/1 | 待填 | 未测 | 未测 | 未测 |

## 最终解释

写明最先发现的三个错误及其根因；为什么 copy/reduce 的 backward 不对称；为什么 DP 平均不能和 TP SUM 混用；PP 为何必须传 activation gradient；性能瓶颈来自哪里、哪些结论有 profile 支持。

报告里没有实际测量的条目保持“未测”。单纯 loss 下降、单纯程序不崩溃，以及正确性测试通过，都不能独立证明高性能。
