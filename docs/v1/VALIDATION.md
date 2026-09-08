# 交付验证记录

验证日期：2026-09-08。

## 必须区分的两种状态

**交付的是带 20 个未完成核心 TODO 的 starter 仓库，不是已经补完的训练框架。**

为确认框架接口、分片布局和验收脚本实际可用，另建了一份临时副本，填入验证实现后进行下列测试。临时实现不在交付压缩包中，也不会被 student 模式隐式调用。用户自己的实现仍必须独立通过验收，不能沿用这里的结论。

## 环境

Linux；Python 3.13.5；PyTorch 2.10.0+cpu；pytest 9.0.2。没有可用 CUDA GPU。多进程使用 torchrun + Gloo，OMP/torch CPU 线程数为 1。

## 原样 starter 的验证

- `python -m pytest -q`：**16 passed, 7 skipped**。7 项是显式选择才启用的学生练习，并非失败被伪装成通过。
- `python -m compileall -q megatron_mini tools tests`：通过。
- `pip install -e . --no-deps --no-build-isolation`：成功建立并安装 editable package；使用环境中已存在的 torch/pytest。
- `python -m megatron_mini.tasks`：识别 20/20 个真实占位符。
- `python -m megatron_mini.train --mode student --steps 1`：在进入通信之前按设计拒绝运行，列出当前路径所需 TODO。
- 完整 dense baseline CPU 训练 20 步成功。合成任务第 0 步 loss 为 **4.266984**，第 19 步为 **2.821514**；不同步样本并不完全相同，此记录只用于 smoke test，不代表严格单调下降或语言能力。

## 临时填充副本的验证

`python -m pytest -q --exercises`：**23 passed**。

双进程独立检查覆盖 group SUM、TP Linear 前向/输入梯度/权重梯度、Embedding mask 梯度、词表 CE 的普通/极端 logits、完整 TP GPT 前后向。

step 测试默认进行 **连续 3 次 AdamW 更新**，比较全局 mean loss、本地 parameter shard 的 gradient 和更新后的参数；带 `--clip 0.25` 的用例同时检查 logical global gradient norm。PP 还包含单 microbatch，以及四个 stage 仅一个 microbatch 的边界。

已经检查 TP、DP、PP 的双进程模式，TP×PP、TP×DP、PP×DP 的四进程组合，以及 **真实八个 CPU 进程的 TP2×PP2×DP2**。这不是只验证八 rank 的算术分组。

checkpoint 验收覆盖 TP2、PP2 和 TP2×PP2×DP2：保存后恢复的下一步 loss/更新/Adam moments/CPU RNG/data cursor 与连续执行一致。CLI 另实际跑过 `steps=2` 保存再 `steps=4` 恢复，日志从 step 2 接续，没有重放 step 0/1。

以下是已执行检查的结果摘要。`max_abs_error` 是对应 case 中所有比较张量的最大绝对差，不是专指 loss；`tolerance_ratio <= 1` 表示通过设定的逐元素 atol/rtol。

```text
PASS case=checkpoint tp=2 pp=2 dp=2 max_abs_error=0.000e+00 tolerance_ratio=0.000e+00
PASS case=step tp=1 pp=1 dp=2 max_abs_error=4.768e-07 tolerance_ratio=1.250e-02
PASS case=step tp=1 pp=2 dp=1 max_abs_error=0.000e+00 tolerance_ratio=0.000e+00
PASS case=checkpoint tp=1 pp=2 dp=1 max_abs_error=0.000e+00 tolerance_ratio=0.000e+00
PASS case=step tp=1 pp=2 dp=1 max_abs_error=4.768e-07 tolerance_ratio=8.150e-03
PASS case=step tp=1 pp=2 dp=2 max_abs_error=4.768e-07 tolerance_ratio=1.250e-02
PASS case=step tp=1 pp=4 dp=1 max_abs_error=0.000e+00 tolerance_ratio=0.000e+00
PASS case=checkpoint tp=2 pp=1 dp=1 max_abs_error=0.000e+00 tolerance_ratio=0.000e+00
PASS case=step tp=2 pp=1 dp=2 max_abs_error=7.246e-07 tolerance_ratio=3.417e-02
PASS case=step tp=2 pp=2 dp=1 max_abs_error=4.247e-07 tolerance_ratio=1.746e-02
PASS case=step tp=2 pp=2 dp=2 max_abs_error=7.246e-07 tolerance_ratio=3.417e-02
PASS case=embedding tp=2 pp=1 dp=1 max_abs_error=0.000e+00 tolerance_ratio=0.000e+00
PASS case=groups tp=2 pp=1 dp=1 max_abs_error=0.000e+00 tolerance_ratio=0.000e+00
PASS case=linear tp=2 pp=1 dp=1 max_abs_error=1.421e-14 tolerance_ratio=3.959e-10
PASS case=loss tp=2 pp=1 dp=1 max_abs_error=5.551e-17 tolerance_ratio=4.482e-09
PASS case=model tp=2 pp=1 dp=1 max_abs_error=4.768e-07 tolerance_ratio=4.416e-02
```

## 未验证与不承诺

CUDA/NCCL 路径没有实机测试；没有 GPU benchmark、显存或扩展性性能结论。CPU 验收不能代替 GPU 后端的正确性和死锁测试。

没有 BF16/FP16/FP8、SP/CP/EP/MoE、1F1B、异步通信、跨拓扑 checkpoint、真实文本变长数据和多机性能验证。依赖声明的整个版本范围也不是兼容性测试矩阵。

数值测试覆盖 tiny configuration 及所列边界；不构成所有形状、硬件与用户填充实现的形式化正确性保证。
