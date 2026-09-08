# 分布式错误定位手册

先固定 tiny config、FP32、同一 seed 和同一 global batch。按“单层 forward -> 单层 backward -> 整模型 -> step -> mixed parallel”顺序排查，不一上来调整学习率掩盖错误。

## 程序启动就退出

出现 `Txx 尚未完成` 或“所需练习尚未实现”是预期的占位符检查，不是环境异常。按 TASKS 补齐当前路径需要的编号。`python -m megatron_mini.tasks` 只扫描真实 `todo(...)` 调用，不根据 README 的复选框判断进度。

`WORLD_SIZE != TP*PP*DP` 表示启动进程数和并行参数不一致。这里 DP 不自动推断，例如四进程 TP2×DP2 需要同时写 `--tp 2 --dp 2`。

`No module named tools` 时确认是在仓库根目录运行 `python -m tools.check_distributed`；tools 是仓库本地的测试驱动，不是安装到系统的通用包。

## 一直卡住

先检查 T02 的纯 Python group-plan 测试，再检查各 rank 是否进入同样顺序的 collective。进入 Gloo/NCCL 之前的配置错误尽量在所有 rank 同步地暴露；修改函数后出现语法错误也应先通过 compileall 找到。

```bash
python -m compileall -q megatron_mini tools tests
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nproc_per_node=2   -m tools.check_distributed --case groups --tp 2
```

NCCL 环境另可加 `NCCL_DEBUG=INFO`。环境变量定义与后端约束见 SOURCES 的 [S8]。不要在不同 rank 使用不同的网络接口配置。

PP 检查同一条链上的 microbatch 顺序：forward 全部升序，backward 全部倒序。send/recv 的 source/destination 是 global rank，消息 shape 和 dtype 必须完全一致。当前主线不使用 tags，而是依靠确定顺序。

不要为了“修好”挂起随意加入 barrier。barrier 不会补出缺失的 send，也不会纠正发错 rank。先定位哪个 rank 等待哪个 rank，再检查逻辑依赖。

## Forward 对，但 backward 错

| 现象 | 首先检查 |
|---|---|
| Column 权重梯度对、输入梯度小一截 | T03 是否遗漏 TP SUM |
| Row 后的梯度大约多 TP 倍 | `_Reduce.backward` 是否被错误改成 all-reduce |
| Row 输出偏移一个固定量 | bias 是否在 reduce 前重复加入 |
| Embedding 某些词的梯度异常 | 域外 token 的安全 index 和 mask 是否同时正确 |
| CE value 看似正常、训练不对 | target ownership、upstream grad、是否重复 mean |
| TP1 对、TP2 attention 错 | local_heads、reshape、head_dim、output projection 输入布局 |
| 多 microbatch 梯度大 M 倍 | loss 没除 M |
| 多 microbatch 梯度小 M 倍 | 多处重复除 M |
| DP 后梯度不符合 global batch | DP group 是否固定 PP/TP，数据是否确实不同 |
| PP 首 stage 没有梯度 | 收到的 activation 是否叶子、是否保留 output 图、是否发送 dInput |
| Clip 打开才出错 | replicated 参数是否重复计数，是否错误在 DP 上 SUM |

## 梯度对，但 AdamW 更新不完全一致

先比较 gradient 再比较 parameter。理论上为零的梯度（例如某些 attention key bias）也可能有微小浮点舍入噪声，而 Adam 的分母会放大这类差异。测试 driver 用 `eps=1e-6` 和明确的 atol/rtol，避免把这个数值现象误判为通信错误。

不能一发现不匹配就无限放大 tolerance。先记录最大绝对误差与归一化 tolerance ratio，定位到参数名；只有 scale-independent 的微小误差才考虑浮点解释。TP/DP/M 整数倍误差通常是语义错误。

默认训练 CLI 使用普通 AdamW 默认 eps，验收 driver 的 eps 是为数值对照显式设置的；两者不应用同一条日志做严格逐步轨迹比较。验收中的 dense/student 使用完全相同 optimizer 参数。

## Checkpoint 恢复后 loss 不一样

确认存的是 `next_step` 而非刚完成的 step。确认 AdamW 的 step、exp_avg、exp_avg_sq 都恢复。确认没有重新开始数据，也没有改变 microbatch 大小/数量。

读取 tensor 在 CPU 上再恢复；torch CPU RNG state 必须保持 CPU byte tensor。CUDA 模式恢复当前设备的 RNG。只存 weight 不是 resume；只保存 rank0 对 PP/TP shard 也不够。

CPU/CUDA 设备类型不同、模型配置不同、TP/PP/DP 不同以及 CLI run_config 不同会被拒绝。这是本周实现边界，不是要在 loading 函数里悄悄放宽的限制。

## 性能看起来很差

这是未融合、阻塞通信、每参数 DP all-reduce、静态 GPipe 的教学框架。tiny 模型很容易被 Python、kernel launch、小通信包和计时边界主导，不应期待并行度越大就越快。

先确认正确性与真实 GPU 环境，再测更合理的工作负载。不要用 CPU Gloo 数值测试里的 tokens/s 估计 NCCL/GPU 性能，也不要把 TP/PP ranks 算成额外样本导致吞吐翻倍。
