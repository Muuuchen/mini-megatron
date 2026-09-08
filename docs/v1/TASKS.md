# 20 个核心 TODO 的实现契约

代码中的 `Txx` 与这里一一对应。函数签名、成员字段、数据布局和模块依赖已固定，优先在给定实现点补代码，不要为了绕过某项练习调用完整 dense 模型。提交前应删除相应 `todo("Txx")`，但保留任务编号和关键契约注释。

所有多进程命令都需在仓库根目录运行；先完成依赖任务。具体每天运行哪些命令见 [WEEK_PLAN.md](WEEK_PLAN.md)。

## T01 · next-token mean CE

位置：`training/loss.py::causal_lm_loss`。输入 `[B,S,V]` logits 和 `[B,S]` 已 shift 的 labels，输出有计算图的标量。考察 reshape、reduction 和 target 对齐，不要求手写 softmax。用 `F.cross_entropy`，禁止 `.item()` 和二次 shift。

验收：`pytest -q --exercises tests/test_exercises.py -k token_loss`。不仅值相同，对 logits 的梯度也相同。

## T02 · 进程组规划

位置：`distributed/topology.py::plan_groups`。输出整个世界的 `GroupSpec` 列表，不接当前 rank。按 TP -> PP -> DP 的固定顺序枚举；每类固定坐标自然升序；包含 singleton 组。

先写八个 rank 的期望分组再实现循环。`ParallelDims.rank/coords` 已完成。不能把 group creation 藏到各 rank 的条件分支里。

验收：`-k group_plan`，再 `tools.check_distributed --case groups --tp 2`。纯算术测试覆盖了 2×2×2、2×3×2 与退化维度。

## T03 · copy 的 backward

位置：`tensor_parallel/mappings.py::_Copy.backward`。forward 已是 identity。对各 TP 分支贡献的 dX 做 SUM，返回 `(grad_x, None)`。不能除 TP；复制传入的 grad 后再通信，避免破坏其他 autograd 消费者。

依赖 T02；通过 T05 触发。验收 `--case linear --tp 2` 的 column input gradient。

## T04 · reduce 的 forward

位置：`tensor_parallel/mappings.py::_Reduce.forward`。将各 rank 的 partial output 做 TP SUM，返回新的完整逻辑输出。backward 的 identity 已提供，不需要再添一遍 all-reduce。

依赖 T02；通过 T06/T07 触发。验收 `--case linear/embedding --tp 2`。注意只在 TP group 上通信。

## T05 · ColumnParallelLinear

位置：`tensor_parallel/layers.py::ColumnParallelLinear.forward`。成员 weight/bias 已按输出维分片。输入复制的 `[B,S,Hin]` 经 copy mapping 后进行本地 Linear，输出 `[B,S,Hout/TP]`。

不 gather 输出，不重新切 weight，不自行平均梯度。考察“为什么普通 F.linear 的 local dX 不完整”。验收 linear 的 output、dX、dW、db。

## T06 · RowParallelLinear

位置：`tensor_parallel/layers.py::RowParallelLinear.forward`。输入已经是 `[B,S,Hin/TP]`，weight 已按输入维分片。本地 Linear 不加 bias，partial result 经过 reduce mapping 后再加复制 bias。

不要重复 split 输入。测试使用非零 bias，能抓住 bias 被加 TP 遍的问题。验收 linear 的 output、local dX、local dW、replicated db。

## T07 · VocabParallelEmbedding

位置：`tensor_parallel/layers.py::VocabParallelEmbedding.forward`。`start/end` 是当前 rank 的全局 token 范围。先判断归属，再安全映射域外 token 到本地 0，做 embedding 后把域外向量清零，最后 reduce。

域外索引必须先变安全值再查表；不能依赖 negative indexing；mask 也要正确阻止域外 token 对第 0 行权重产生梯度。验收 `--case embedding --tp 2`。

## T08 · vocab CE forward

位置：`tensor_parallel/loss.py::_VocabCE.forward`。输入 local logits，输出每个 token 的 loss，不做 mean。通过 TP MAX 获得稳定偏移，通过 TP SUM 得到分母及目标 logit。保留原始输入不变。

自己决定 ctx 保存的 tensor 组织，但与 T09 一致。建议保存 local probabilities、安全 local target index、ownership mask。不能 all-gather 全词表，也不能只对本地 logits 做普通 CE。

验收 `-k single_rank_vocab_loss` 和 `--case loss --tp 2`；包括大 logits、边界 targets。float64 数值单测不应被无条件 `.float()` 降精度。

## T09 · vocab CE backward

位置：`tensor_parallel/loss.py::_VocabCE.backward`。局部 softmax 减当前 shard 的局部 one-hot，再乘 `[B,S]` upstream gradient。输出 local logits gradient，targets 与 mesh 返回 None。

不重复做 mean，不乘除 TP，只有 target 的归属 shard 减 1。测试采用任意 upstream gradient，不允许只写一个默认 mean 的特例。验收同 T08。

## T10 · head-parallel attention

位置：`model/parallel.py::ParallelAttention.forward`。Q/K/V 列并行后转为 `[B,local_heads,S,head_dim]`，调用已完成的 `causal_attention`，再转回 `[B,S,H/TP]` 并经 Row projection。

不手写 CUDA kernel，不 gather heads，不把总 heads 当本地 heads。这里 Q/K/V 是三个独立矩阵，没有 fused QKV 的跨段切片问题。先单进程模型验收，再 Day 4 跑 `--case model --tp 2`。

## T11 · TP MLP

位置：`model/parallel.py::ParallelMLP.forward`。连接 column Linear、`F.gelu` 和 row Linear。激活的局部 `[B,S,F/TP]` 不离开本 rank。

和 dense 使用同一种 GELU，不更换成 tanh approximation 或 SwiGLU。通过模型 logits/grad 检查，不以 loss 下降代替等价性。

## T12 · pre-norm block

位置：`model/parallel.py::ParallelBlock.forward`。先 attention 的 pre-norm/residual，再 MLP 的 pre-norm/residual。第二段处理的是更新后的 x。

LayerNorm 参数复制，norm 的 hidden 维完整；本实验没有 sequence parallelism。不要随意添加 TP gradient sync，它已经由正确的上下游 mapping 产生完整梯度。

## T13 · 本地 GPT stage

位置：`model/parallel.py::GPTStage.forward`。首 stage 做 token+position embedding；每个 stage 遍历自己的 blocks；末 stage 加 final norm 和词表分片 head。非末输出 hidden，末输出 local logits。

首 stage 接 int64 tokens，其余接 FP32 activation。PP=1 同时 first/last，两端都要执行。成员分配、层切分和全局参数名已完成；不得再创建完整模型绕过 PP。

验收 `-k student_model`；补完 T16/T17 后在 PP step 测试中同时验收。

## T14 · DP 梯度平均

位置：`training/grad.py::average_dp_gradients`。在当前 step 所有 backward 结束后，对每个本地参数 grad 做 DP SUM / DP。`dp=1` 已直接返回。

本 lab 不允许 unused parameters，grad=None 应报错；不能在不一致的 skip 分支里漏掉 collective。禁止 WORLD average，否则会混入模型的其他 shard。不要实现每 microbatch 都同步一遍的慢路径。

验收 `--case step --dp 2`，再可选 TP2×DP2。比较 global batch 的 dense 梯度和连续更新。

## T15 · 无 PP 的 microbatch 累积

位置：`training/schedules.py::forward_backward_no_pipeline`。调度每个 microbatch forward 和立即 backward，loss 除 M。返回 detached 的本 DP rank 平均 loss。

调用者已 zero_grad，调用者会 DP average、clip 和 optimizer.step；这里都不要做。不能每个 microbatch 覆盖掉前一份梯度，也不要攒完全部图才反向。

验收 `-k student_gradient`；随后单进程 student 训练和 TP/DP step 检查。

## T16 · GPipe forward 与图缓存

位置：`pipeline_parallel/schedules.py::gpipe_forward`。按 microbatch 升序执行：首 stage 取 tokens，其余 recv 一个叶子 activation；执行本地 stage；非末 send output，末 stage 计算 mean token loss。返回升序 Record 列表。

缓存的 output 不能 detach；通信的 payload 应 detach。中间 stage 保存输入叶子，以便 T17 读取其 grad。不要用全局列表跨 step 保存图；不要提前 backward。

验收依赖 T17，运行 PP2，分别 M=1/M=3；多资源时 TP2×PP2。

## T17 · GPipe 反向 drain

位置：`pipeline_parallel/schedules.py::gpipe_backward`。倒序处理 Records。末 stage 对 loss/M 反向；其他接收下一 stage 的 dOutput 并执行 tensor backward；非首 stage 发送自身 input_activation.grad。

只在末 stage 除 M 一次。非末日志返回 device 上的 0 标量；末 stage 返回 detached 平均 loss。结束清空 records。不能 `retain_graph=True` 掩盖图生命周期错误。

验收除 PP2 外，建议用 PP4/M1 验证 M<PP。所有 stage 参数都要与 dense 切片对齐。

## T18 · 逻辑模型 global grad norm

位置：`training/grad.py::clip_grad_norm`。DP average 之后，累计本地 gradient squared norm；所有 shard 计入，TP replicated 参数只在 t=0 计入。FP64 累加，先 TP SUM 后 PP SUM，开根号得到裁剪前 norm。

所有 rank 使用相同 factor，所有本地 grad 都裁剪。不要把“只计数一次”误写成“只更新 t=0 副本”。不做 DP SUM。

验收 `-k clip_norm`，以及 `--case step --clip 0.25` 的 TP、PP 和混合配置。

## T19 · 保存同拓扑 checkpoint

位置：`checkpoint.py::save_checkpoint`。保存 meta/model/optimizer/next_step/cpu_rng/cuda_rng 六类内容，使用约定 key。目录路径与 per-rank filename helper 已提供。

`next_step` 表示下一个数据 step；保存必须在 optimizer.step 完成后。先临时文件、后 os.replace；不同 rank 写不同文件。没有跨 rank manifest 提交事务，不宣称支持中途崩溃恢复。

验收可不执行模型 forward 的 `-k checkpoint` 单测，随后完整 distributed resume 测试。

## T20 · 恢复与继续训练

位置：`checkpoint.py::load_checkpoint`。先读到 CPU 并验证 metadata，再恢复 model、optimizer、CPU/当前 CUDA RNG，返回 next_step。优先 `weights_only=True`，只加载本项目产生的可信文件。

CLI 通过 run_config 约束 microbatch 设置、学习率、clip 设置。改变模型/并行度/设备类型也应拒绝，而不是勉强加载。只恢复模型会丢失 Adam moments，不是等价 resume。

验收 `--case checkpoint --tp 2` 和 `--pp 2`；测试将恢复后的下一步与 uninterrupted training 进行比较。
