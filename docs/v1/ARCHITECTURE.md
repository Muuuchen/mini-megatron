# 架构与语义契约

## 1. 如何从 Megatron 拆出这一周的主线

本仓库沿用“训练入口 + 可组合核心模块”的分层思路，但不复刻 Megatron 的所有类、参数和生产特性。原项目区分 Megatron-LM 训练参考入口和 Megatron Core 构件；原始出处与源码对应位置见 [SOURCES.md](SOURCES.md) 的 [S3–S7]。

我们的依赖是：`train -> engine -> schedule -> GPTStage -> TP layers -> communication mappings -> Mesh`。loss 位于模型之外；checkpoint 位于 step 之外；P2P 只搬数据，不替 scheduler 管理图。底层模块不反向 import 训练入口。

`DenseGPT` 是没有分布式逻辑的数值 oracle。`GPTStage` 的参数名与 dense 对应，block 保留全局层号，因此切片和数值核验不需要把本地第 0 层误认为全模型第 0 层。

## 2. Rank 网格

我们固定 `[DP, PP, TP]` 布局：

```text
rank = (dp_rank * pp_size + pp_rank) * tp_size + tp_rank
world_size = dp_size * pp_size * tp_size
```

TP 组固定 DP、PP，改变 TP；PP 组固定 DP、TP，改变 PP；DP 组固定 PP、TP，改变 DP。这个布局是本实验的选择，不声称是 Megatron 唯一或固定的 rank ordering。

TP=PP=DP=2 的例子：

| 类型 | 组 |
|---|---|
| TP | (0,1), (2,3), (4,5), (6,7) |
| PP | (0,2), (1,3), (4,6), (5,7) |
| DP | (0,4), (1,5), (2,6), (3,7) |

`plan_groups` 在每个 rank 返回相同的完整列表。`initialize` 才依据列表创建真实 ProcessGroup。非成员也参与对应 `new_group` 的全局调用顺序；这是 PyTorch 的调用约束，而不是本框架的特殊规定，见 [S8]。

`Mesh.peer(p)` 返回 **global rank**。不能把 PP 逻辑序号直接交给 `dist.send(dst=...)`。

## 3. 张量分片契约

记局部 microbatch 为 B，序列长度为 S，hidden 为 H，FFN 中间维为 F，词表为 V，TP 度为 T。

| 边界/参数 | 本地布局 | 复制或分片 |
|---|---|---|
| token IDs | [B,S] | TP/PP 同一 DP 样本 |
| token embedding weight | [V/T,H] | 按词表分片 |
| embedding 输出、残差、LayerNorm 输入 | [B,S,H] | TP 内复制 |
| Q/K/V weight | [H/T,H] | 各自按输出特征分片 |
| Q/K/V 激活 | [B,S,H/T] | 按完整 heads 分片 |
| attention context | [B,heads/T,S,D] | heads 分片，D 不变 |
| attention output projection weight | [H,H/T] | 按输入特征分片 |
| MLP fc1 weight | [F/T,H] | 输出分片 |
| MLP fc2 weight | [H,F/T] | 输入分片 |
| LM head weight | [V/T,H] | 输出词表分片 |
| LM logits | [B,S,V/T] | 末 stage 的各 TP rank 一片 |
| PP 边界激活及其梯度 | [B,S,H] | 每条 TP lane 发送一个完整边界张量 |

以上 weight shape 遵循 PyTorch `[out,in]`，不要用论文里 `Y=XW` 的列/行记号直接推断维度编号。层并行的原始设计与 API 可读 [S4–S5]。

LayerNorm、位置 embedding、Row bias 是 TP replicated 参数。`_shard_dim=None` 标记复制；`0/1` 标记参数实际分片方向。这个 metadata 只服务本仓库，不是 PyTorch 内建属性。

## 4. 为什么两个 autograd mapping 不对称

`copy_to_tp`: forward identity，backward TP SUM。一个复制输入参与多个不同的权重分片，局部 dX 是不同分支的贡献，必须相加。

`reduce_from_tp`: forward TP SUM，backward identity。forward 已把局部乘积合为完整逻辑输出，后续各 rank 执行同一逻辑 loss；不能把复制的上游梯度再加 T 次。

这不是“所有 all-reduce 的数学导数都是 identity”，而是 Megatron 风格的逻辑张量分布与 loss 执行契约，[S6] 的 mapping 明确区分这两种情况。脱离这个执行契约，不能复用同样的 backward。

因此 MLP 是 `copy -> local GEMM -> GELU -> local GEMM -> reduce`。Row 的 replicated bias 加在 reduce 之后。attention 也遵守同样的输入合并和输出归约思路。

## 5. 词表并行 loss

对某 token 的完整 logits z，目标为 y：

```text
m = max over all vocab shards(z)
s = sum over all vocab shards(exp(z - m))
loss = log(s) + m - z[y]
local_dz = local_softmax - local_one_hot(y)
```

只需要对逐 token 的 max、分母、目标 logit 做规约，通信张量是 `[B,S]`。不用 gather `[B,S,V]`。max 使用 MAX，不是 SUM；目标 logit 只有归属 shard 贡献。

forward 返回逐 token loss；外层 `.mean()` 负责 token mean；再由 scheduler 除 microbatch 数。backward 只乘传入的 upstream gradient，不额外除 TP。数值稳定测试同时覆盖常规 logits 和大幅值 logits。

## 6. global batch 与梯度缩放

```text
local_batch_per_DP_replica = micro_batch * num_microbatches
global_batch = micro_batch * num_microbatches * dp_size
```

TP 和 PP 不增加有效样本数。固定、无 padding 的 S 下：

```text
L_global = (1 / DP) * sum_d [ (1 / M) * sum_m L_token_mean(d,m) ]
```

实现对应：每个 microbatch loss 除 M；DP gradient SUM 后除 DP；PP 只有最后 stage 进行 loss/M，其余 stage 只传播已经缩放的梯度。一个更新只调用一次 `optimizer.step()`。

`reported_loss` 是 detached 的日志规约：非末 PP stage 提供 0，PP SUM 后取得完整 loss，再 DP 平均。不对 TP replicas 求和；不拿经过全局日志规约的标量反向传播。

扩展到不等长序列、padding mask、不等 microbatch 或 dropped tokens 后，以上简单 mean-of-means 可能错误；应按真实 token count 推导分母，不能直接复用当前公式。

## 7. Pipeline 调度与计算图

每个 stage 的局部顺序是：

```text
forward:  F0 -> F1 -> ... -> F(M-1)
backward: B(M-1) -> ... -> B1 -> B0
```

不同 stage 可以在不同 microbatch 上工作；阻塞传输只保证相邻发送/接收配对，不是把所有 ranks 放进一条全局串行循环。此处仍不实现通信与计算异步重叠。

一次本地 forward 产生 `Record(input_activation, output, loss)`。前一个进程的 autograd 图不会自动穿过网络。中间 stage 的 `recv_forward()` 分配新的叶子 tensor，设置 `requires_grad=True`；本地 backward 后读取这个叶子的 `.grad`，发送给前一 stage。

前一 stage 保留自己的原始 output（带 grad_fn），收到梯度后执行 `torch.autograd.backward(output, grad)`。只有发送 payload detach，不能把缓存 output detach；只有 loss 所在 stage调用标量 backward。

没有 `retain_graph=True`，每个图只反向一次。反向结束清空 records，防止跨 optimizer step 泄漏激活引用。GPipe-style 这里指 fill-drain 调度；没有实现原论文 [S9] 的 activation rematerialization。

M=1 也合法；M 不必大于等于 PP。只是流水线可用并发不足，不应因为这个边界出现死锁或错误梯度。

## 8. TP/PP-aware gradient clipping

DP 梯度平均后，各个 DP 副本已经表示同一个逻辑模型梯度。每个 rank 对本地平方和计数时，分片参数全部计入；TP replicated 参数仅在 `tp_rank=0` 计入。

然后 TP SUM 合并一个 stage 的完整梯度平方和，再 PP SUM 合并不同 stage 的参数。不要对 DP SUM，否则会多计 DP 份。

同一个 norm 用来计算所有 ranks 相同的裁剪系数。复制参数虽只计数一次，但每份参数副本的 grad 都必须乘该系数。测试在低于实际 norm 的阈值上触发真实裁剪，而不是只测不裁剪的分支。

## 9. 初始化与恢复

所有训练模式从同 seed 的 dense 参数开始，再依据 `_shard_dim` 和全局层名取片。独立初始化一个小 weight 不是完整 weight 分片后的相同随机数，不能用“seed 一样”代替参数切片一致性。

初始化中的 dense 模型只在 CPU 暂存；实际 stage 只分配自己拥有的层。分布式验收脚本另外持有 dense oracle 是测试开销，不是 student 训练架构。

checkpoint 必须存模型、AdamW 状态、next_step 和使用到的 RNG。CLI 的 run_config 记录 microbatch 大小/数量、学习率、clip 设置；恢复检查这些值与模型配置、并行度和设备类型一致。`--steps` 不在约束里，因为恢复后的总步数需要增大。

只做同拓扑、完整 step 边界的恢复，不保证进程中途崩溃后的跨 rank 一致提交。中途打断产生的部分目录应丢弃，不要声称这是 fault-tolerant checkpoint 系统。
