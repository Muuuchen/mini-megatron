# v3新增任务索引

T01–T60沿用下文；新增 **T61–T72完整契约在[MEMORY_TASKS.md](MEMORY_TASKS.md)**。SAC替代重复全unit重算，不自动叠加T33。

# 72 个核心 TODO · v2

T01–T20保留基础路径；T21–T60扩展完整机制。先看函数docstring，再按WEEK_PLAN做独立case，最后接T55。
`lab --list`显示case需要的最小任务集合，不强制先完成全部v1。
**独立case提供了数值oracle；T40/T55/T59的真实模型接线仍需按INTEGRATION编写集成断言。**


# T01–T20：基础训练与TP/DP/GPipe契约

代码中的 `Txx` 与这里一一对应。函数签名、成员字段、数据布局和模块依赖已固定，优先在给定实现点补代码，不要为了绕过某项练习调用完整 dense 模型。提交前应删除相应 `todo("Txx")`，但保留任务编号和关键契约注释。

以下描述适用于v1基础路径；具体执行日期以v2 WEEK_PLAN.md为准。

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


# T21–T60：新增机制


## T21 · DP/TP/PP/CP/EP/EDP group plan

Day 1；位置 `distributed/topology_v2.py`；依赖：无其他学生任务。


`plan_groups_v2`（生成时行号 38）：

T21: 返回全世界相同顺序的 GroupSpec，包含 singleton。
顺序 tp,pp,cp,dp,ep,edp,dp_cp,edp_cp。
tp/pp/cp/dp: 只变化对应坐标。ep: 固定 d//ep，变化 d%ep。
edp: 固定 d%ep，变化 d//ep。dp_cp: 固定(p,t)，变化(d,c)。
edp_cp: 固定(p,t,d%ep)，变化(d//ep,c)。
每类按 ranks tuple 排序。FSDP 在 dp/edp 上分片，CP 先归约贡献。
验收: test_v2_exercises::test_mesh_groups；不能每个 rank 自己 new_group。


独立验收：`python -m megatron_mini.lab --case topology`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T22 · SP gather backward

Day 2；位置 `sequence_parallel/mappings.py`；依赖：无其他学生任务。


`backward`（生成时行号 11）：

T22: [B,S,H] partial dX -> SUM+sequence shard，不能只切片。


独立验收：`python -m megatron_mini.lab --case sp`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T23 · SP reduce-scatter forward

Day 2；位置 `sequence_parallel/mappings.py`；依赖：无其他学生任务。


`forward`（生成时行号 17）：

T23: partial [B,S,H] -> SUM 后 [B,S/TP,H]，保存 axis。


独立验收：`python -m megatron_mini.lab --case sp`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T24 · SP MLP/residual layout

Day 2；位置 `sequence_parallel/block.py`；依赖：T22, T23。


`sp_mlp`（生成时行号 4）：

T24: x[B,S/TP,H], w1[F/TP,H], w2[H,F/TP] -> 与 x 同形。
顺序: local norm -> AG -> local F.linear(w1,b1) -> GELU ->
local F.linear(w2,None) -> RS -> +b2 -> +local residual。
不可直接调用旧 ColumnParallelLinear: 它会重复 all-reduce dX。
b2 在 RS 后添加，只添加一次。norm/b2 的梯度在 T25 做 TP SUM。
验收: lab sp，前向、local dX、两种分片权重及复制 LN/bias 梯度。


独立验收：`python -m megatron_mini.lab --case sp`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T25 · SP replicated gradient SUM

Day 2；位置 `sequence_parallel/block.py`；依赖：T24。


`sync_sp_replicated`（生成时行号 14）：

T25: 只同步 sequence-local norm/bias 参数的 grad，SUM 不平均。
先有完整名称/角色清单再遍历，不要对 TP-sharded GEMM 参数做 SUM。
同步在所有 microbatch 完成后，和 DP reduction 是不同的维度。


独立验收：`python -m megatron_mini.lab --case sp`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T26 · CP KV gather backward

Day 3；位置 `context_parallel/attention.py`；依赖：无其他学生任务。


`backward`（生成时行号 10）：

T26: 不同 query shard 对全局 KV 的梯度贡献 SUM+scatter，dim=2。


独立验收：`python -m megatron_mini.lab --case cp`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T27 · CP global-mask attention

Day 3；位置 `context_parallel/attention.py`；依赖：T26。


`gather_attention`（生成时行号 14）：

T27: q/k/v[B,heads,S/CP,D] -> local output，按 contiguous 序列切分。
用 _GatherKV 只聚合 K/V，Q 不聚合。mask 用全局 query offset:
qpos=rank*S_local+arange(S_local)，允许 kpos<=qpos。
logits FP32（float64 测试保持 float64），softmax 后乘 V。
这里物化完整 KV，是 CP 的 correctness baseline，不宣称 ring 内存收益。
验收: lab cp，跨 rank causal mask、dQ/dK/dV。


独立验收：`python -m megatron_mini.lab --case cp`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T28 · Ring online softmax merge

Day 3；位置 `context_parallel/ring.py`；依赖：无其他学生任务。


`online_update`（生成时行号 3）：

T28: 一块 scores[...,Q,K] 的 online softmax 更新，返回(m,l,o)。
m/l:[...,Q,1]；o 是未归一化分子[...,Q,D]；初态 -inf/0/0。
valid 可广播到 scores，masked scores 视为 -inf。
合并旧、新 max，指数重标定 l 和 o，禁止每个 KV block 各算 softmax 再加。
全 masked 的行必须保持 0 分子/分母、不产生 NaN（-inf-(-inf)）。
最后 output=o/l，仅在确有可见 key 时相除。


独立验收：`python -m megatron_mini.lab --case ring`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T29 · Ring KV circulation and backward

Day 3；位置 `context_parallel/ring.py`；依赖：T28。


`ring_attention`（生成时行号 13）：

T29: 真正的 ring P2P + 自定义 backward。API 同 gather_attention。
每轮只持有本地与接收 KV buffer，owner ID 随环转动，用全局位置 mask。
forward 用 T28；backward 再循环 KV，dQ 本地累加、dK/dV 回到 owner。
保存 q/local kv/output/logsumexp；不能保存全部 rank 的 KV。
可按 block 重算 scores，不要求手写 FlashAttention CUDA kernel。
验收: lab ring 对 gather baseline 的值/梯度；双 buffer 生命周期；CP2/4。
高阶导数不在契约内。CPU只能验语义，真正内存/overlap必须测 CUDA。


独立验收：`python -m megatron_mini.lab --case ring`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T30 · Noninterleaved 1F1B event plan

Day 4；位置 `pipeline_parallel/planner.py`；依赖：无其他学生任务。


`plan_1f1b`（生成时行号 10）：

T30: 非交错 per-stage event 列表，支持 M<P。
warmup=min(P-stage-1,M)；先 warmup 次 F；随后 F/B；最后 cooldown B。
FIFO backward ID，不要沿用 GPipe 的 reverse microbatch 顺序。
只能返回 F/B event，通信由独立 executor 管理。
验收: event 覆盖、依赖、live records；P=2/4, M=1/2/8。


独立验收：`python -m megatron_mini.lab --case pipeline`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T31 · Pipeline event executor

Day 4；位置 `pipeline_parallel/executor.py`；依赖：T30。


`execute_events`（生成时行号 15）：

T31: 通用 GPipe/1F1B/VPP executor，records[(mb,chunk)]保存图。
F: receive input leaf -> forward -> 非末 async send detached output。
B: 末 stage(loss/M).backward；非末 recv dout -> autograd.backward；
   非首 send input.grad；删除该 record。只有最后 logical stage 缩放 loss。
在相邻 rank 能都先发的场合不能使用两个阻塞 send；已提交的 send buffer
在 handle 完成前不能释放/重用。最后 io.finish()。返回 detached mean loss。
验收: pipeline lab + recipe pipeline_1f1b，M<P/非零 bias/多次 update。


独立验收：`python -m megatron_mini.lab --case pipeline`、`python -m megatron_mini.lab --case interleaved`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T32 · Virtual/interleaved pipeline plan

Day 4；位置 `pipeline_parallel/planner.py`；依赖：T30。


`plan_interleaved`（生成时行号 19）：

T32: 最小 VPP=2 的 per-physical-stage event 列表。
chunk c 在物理 p 上表示 logical stage c*P+p；跨 chunk 的边界是
p=P-1 -> p=0，而不是两个本地 chunk 连着执行。backward 反向。
可先限制 M 为 P 的正整数倍，其他情况显式 ValueError。
每个(microbatch,chunk)一次 F/B；FIFO 与跨 rank matching 不可混淆。
用 validate_schedule_simulation 做全局依赖/死锁验收再接 executor。


独立验收：`python -m megatron_mini.lab --case interleaved`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T33 · Activation rematerialization

Day 3；位置 `runtime/recompute.py`；依赖：无其他学生任务。


`recomputed`（生成时行号 3）：

T33: enabled 时用 torch.utils.checkpoint.checkpoint(use_reentrant=False)。
preserve_rng_state=True；disabled 直接 module(*args)。不 detach inputs。
验收带 dropout 的 output/gradient/RNG一致性，至少一个输入不 requires_grad。
block 与 attention 两种粒度由构造器选择 wrapper 位置；不是模型存盘。


独立验收：`python -m megatron_mini.lab --case recompute`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T34 · DP bucket last-microbatch ready

Day 4；位置 `runtime/overlap.py`；依赖：无其他学生任务。


`mark_bucket_ready`（生成时行号 23）：

T34: autograd post-accumulate hook 最后一个 microbatch 标记 grad ready。
所有参数都ready时恰好一次launch(buffer)；launch返回Pending。旧梯度不能丢。
每个rank统一bucket定义与launch顺序，未用专家参数要显式zero，不可skip collective。
optimizer.step 前wait所有bucket；不可每microbatch把同一累积grad重复归约。
lab bucket模拟ready时序；CUDA trace才验真正compute/comm overlap。


独立验收：`python -m megatron_mini.lab --case bucket`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T35 · Flat buffer shard layout

Day 5；位置 `optimizer/flat.py`；依赖：无其他学生任务。


`build`（生成时行号 22）：

T35: named_tensors 是有序(name,tensor)列表，记录 shape/offset。
padded_numel=ceil(numel/world_size)*world_size，支持参数跨 shard 边界。
拒绝空参数、重复名字、共享 storage/tied 参数（不静默重复更新）。
输出 FlatLayout；原始 shapes 不包含 padding。


独立验收：`python -m megatron_mini.lab --case flat`、`python -m megatron_mini.lab --case zero1`、`python -m megatron_mini.lab --case zero2`、`python -m megatron_mini.lab --case fsdp`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T36 · ZeRO-1 optimizer sharding

Day 5；位置 `optimizer/zero.py`；依赖：T35。


`step_zero1`（生成时行号 36）：

T36: full grad -> DP allreduce/DP -> local shard.grad -> local AdamW
-> allgather updated local params -> copy_back。每步只更新一次。
先不做 clip，由统一 engine 在正确时间处理。验收3次更新和 moments。


独立验收：`python -m megatron_mini.lab --case zero1`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T37 · ZeRO-2 reduce-scatter gradient

Day 5；位置 `optimizer/zero.py`；依赖：T35。


`step_zero2`（生成时行号 43）：

T37: full/bucket grad -> reduce_scatter /DP -> local AdamW
-> allgather params。禁止对 RS 后 shard 再做一次 DP average。
用全长瞬时 buffer 做数值 baseline，但内存实验必须接 T34 分桶。


独立验收：`python -m megatron_mini.lab --case zero2`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T38 · FSDP materialize/release forward

Day 5；位置 `fsdp/unit.py`；依赖：T35。


`forward`（生成时行号 14）：

T38: gather local_shard 得到 flat full params -> functional_call。
ctx 只保存 x/local_shard、unit和 CPU/CUDA RNG状态，不保存 full flat。
记录 forward 前 RNG；backward重算时恢复但不扰动训练的当前 RNG。
ctx 可保存 autocast配置供反向复现；baseline unit FP32。
full 参数仅存在当前调用中；返回后必须不再被 unit/template/ctx 持有。


独立验收：`python -m megatron_mini.lab --case fsdp`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T39 · FSDP regather/reduce-scatter backward

Day 5；位置 `fsdp/unit.py`；依赖：T38。


`backward`（生成时行号 24）：

T39: 再 gather -> full.detach().requires_grad_()；浮点x detach+require；int64 token输入不求dx。
enable_grad + fork_rng 重算 unit.call(x,full)，用 autograd.grad 得到
dx 与 dflat。先沿 unit.grad_sum_axes SUM，再 RS(dflat)/unit.grad_divisor，
返回(dx,dlocal,None)，整数输入dx=None。默认grad_divisor=axis.size。
x 是各 DP rank 不同样本，dx 不做 DP 平均；dflat 才做平均。
高阶导数不要求；不通过 p.data 替换 shape 欺骗 autograd。
若外层已采用 sum/global_token_count，RS divisor 必须改为1（T49）。


独立验收：`python -m megatron_mini.lab --case fsdp`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T40 · Per-block FSDP units

Day 5；位置 `fsdp/unit.py`；依赖：T39。


`shard_blocks`（生成时行号 61）：

T40: bottom-up 用 FullShardUnit 管理每个 transformer block或MLP block。
剩余 embedding/norm/head 另成 units；只能在结构构造时做，optimizer 后建。
不要将整个 GPT 聚为一个 full unit，否则无法观察层级 live memory 差异。
单元必须是 tensor->tensor；GPT token lookup边界由 T55 adapter 明确处理。


集成验收：按 `docs/INTEGRATION.md` 对应接线步骤执行，不以默认pytest通过代替本任务完成。


## T41 · FSDP next-unit prefetch

Day 5；位置 `runtime/overlap.py`；依赖：无其他学生任务。


`prefetch_next`（生成时行号 32）：

T41: 先 launch next_unit 的参数 gather，执行 current compute，然后
在下一个 unit 消费参数前 wait；最多当前+下一unit full buffers。
API 返回(result, Pending或None)。next=None不能发collective。
本函数教学单元返回的pending由executor交给下一个unit，而非立刻wait。


独立验收：`python -m megatron_mini.lab --case overlap`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T42 · MoE top2 router and balance loss

Day 6；位置 `moe/router.py`；依赖：无其他学生任务。


`topk_route`（生成时行号 3）：

T42: logits[N,E] -> (expert_ids[N,k], weights[N,k], aux_loss)。
在FP32 softmax上topk，再对选中的k个prob归一化；weights保留router梯度。
aux=E*sum(mean_full_probs * (token_assignment_counts/(N*k)))。
离散counts不求导。k=1归一化权重恒1，主loss没有gate梯度是该约定的结果；
因此主课用top2并单测aux梯度。没有capacity/drop/padding：dropless。


独立验收：`python -m megatron_mini.lab --case moe`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T43 · MoE permutation and routing metadata

Day 6；位置 `moe/dispatch.py`；依赖：无其他学生任务。


`pack`（生成时行号 14）：

T43: 重复topk tokens，按owner=floor(expert_id/(E/EP))稳定排序。
返回(packed_x,RoutePlan)。相同owner内部用expert_id排序方便本地执行。
token_ids/slots记录原来token与topk槽位，反向combine不可靠猜。
所有send_splits长度=EP、和=N*k，允许0个token去某个rank。


独立验收：`python -m megatron_mini.lab --case moe`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T44 · Variable AllToAll backward

Day 6；位置 `moe/dispatch.py`；依赖：无其他学生任务。


`backward`（生成时行号 32）：

T44: inverse AllToAll；交换 send/recv split sizes，其他参数无梯度。
返回(grad_input,None,None,None)；不能allreduce，也不除EP。


独立验收：`python -m megatron_mini.lab --case moe`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T45 · MoE weighted unpermutation

Day 6；位置 `moe/dispatch.py`；依赖：T43。


`combine`（生成时行号 47）：

T45: returned顺序与packed_x相同，用weights[token_ids,slots]加权，
然后 index_add/scatter_add 回[N,H]。topk>1不得覆盖写；不detach weights。
参数梯度与router梯度都要对齐，而不是只看token顺序。


独立验收：`python -m megatron_mini.lab --case moe`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T46 · Dense and expert gradient ownership

Day 6；位置 `moe/layer.py`；依赖：T42, T43, T44, T45。


`synchronize_moe_gradients`（生成时行号 33）：

T46: 等长local mean损失约定：router在dense-DP SUM/denseDP。
expert先在EP dispatcher backward已合并不同源token贡献，再在EDP SUM/denseDP
（不是只除EDP，也不在EP组同步不同专家）。denseDP=EP*EDP。
此lab CP=1；CP>1由T49扩为dp_cp与edp_cp。零token专家仍参与。
若外部loss已除global tokens，则两路均只SUM，禁止再平均。


独立验收：`python -m megatron_mini.lab --case moe`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T47 · BF16/FP16 compute policy

Day 3；位置 `runtime/precision.py`；依赖：无其他学生任务。


`autocast_context`（生成时行号 21）：

T47: fp32 nullcontext，bf16/fp16 torch.autocast。FP32 master params/Adam
moments不直接转半精度；loss/grad reduction FP32。FP8显式ValueError。
本课程fp16运行实验要求CUDA；CPU可运行bf16可用算子或测scaler逻辑。
PP激活传输dtype必须与真实output一致，由T59传输协议负责。


独立验收：`python -m megatron_mini.lab --case precision`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T48 · Distributed unscale/overflow agreement

Day 3；位置 `runtime/precision.py`；依赖：无其他学生任务。


`unscale_and_check`（生成时行号 30）：

T48: grad原地除scale一次，检查isfinite。axes上归约 found_inf OR，
返回全局相同的bool；所有rank决定是否跳过这次optimizer.step。
在clip前unscale；整个microbatch窗口scale固定；不得每microbatch update。
axes覆盖共同step的TP/PP/DP/CP/EP成员，哪怕某个rank没有grad也参与。
参数为空时调用者须提供至少一个设备占位tensor，或在runtime提前拒绝。


独立验收：`python -m megatron_mini.lab --case precision`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T49 · Role-aware synchronization and norm

Day 7；位置 `runtime/grad_roles.py`；依赖：T21。


`finalize_gradients`（生成时行号 16）：

T49: 按显式roles归约/缩放，然后unique logical norm与裁剪，返回preclip norm。
每个参数必须有role；SP norm/bias先TP SUM，dense再DP×CP平均；专家只在
EDP×CP上合并，但等长local mean约定的分母仍是denseDP×CP。
FSDP已经RS的grad不重复平均；norm统计则要包含每个owner的所有shards。
count_for_norm仅决定范数计数，不决定是否缩放/更新该参数的副本。
所有norm_axes SUM平方和，最后sqrt；padding必须剔除。不要 blindly WORLD norm。
验收: grad_roles单元测试 + dense_tp_zero2/moe_ep_fsdp的连续update。


独立验收：`python -m megatron_mini.lab --case roles`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T50 · Sharded checkpoint manifest

Day 7；位置 `runtime/checkpoint.py`；依赖：无其他学生任务。


`save_sharded_checkpoint`（生成时行号 14）：

T50: checkpoint v2 payload包含model shards,optimizer moments,step,
mesh/layout/parameter roles,precision/scaler,RNG,data cursor。
每rank atomic replace写shard；确认所有shard写完后，rank0最后提交manifest。
没有manifest/缺任一shard必须拒绝恢复，不能拿半套文件继续训练。
提交后无待通信buffer。这里只保证同topology完整快照，不承诺容错服务。
返回最终manifest路径；load/validate可写成同文件私有helper。


独立验收：`python -m megatron_mini.lab --case checkpoint`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T51 · RMSNorm

Day 1；位置 `model/llama.py`；依赖：无其他学生任务。


`rms_norm`（生成时行号 8）：

T51: 在hidden维计算mean(x^2)，FP32中间值（double测试不降精度），
rsqrt 后还原dtype并乘weight。不是减mean的LayerNorm。


独立验收：`python -m megatron_mini.lab --case model`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T52 · RoPE global positions

Day 1；位置 `model/llama.py`；依赖：无其他学生任务。


`apply_rope`（生成时行号 14）：

T52: x[B,heads,S,D]，positions[S]使用global CP offset。
本课程固定相邻偶奇配对：rotate(x_even,x_odd)，频率base^(-2i/D)。
旋转保持范数；position=0是identity；D必须偶数。禁止不同rank重置positions。


独立验收：`python -m megatron_mini.lab --case model`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T53 · GQA grouped Q-KV mapping

Day 1；位置 `model/llama.py`；依赖：无其他学生任务。


`gqa_attention`（生成时行号 22）：

T53: q[B,Hq,S,D], kv[B,Hkv,S,D]，Hq%Hkv==0。
通过repeat_interleave映射Q组对应KV，然后显式causal attention。
这是dense reference；TP版本还要求本课程Hkv能被TP整除，不做KV复制特例。
验收与逐Q-head选择所属KV的独立公式对齐，含dK/dV累加。


独立验收：`python -m megatron_mini.lab --case model`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T54 · Real text and global token denominator

Day 1；位置 `runtime/text.py`；依赖：无其他学生任务。


`masked_token_loss`（生成时行号 23）：

T54: 返回LossStat。输入各rank的有效token数可不同，不能平均local means。
backward_loss=local有效CE sum / global_valid_count；reported_loss=
global有效CE sum / global_valid_count。只有日志项detach，计数/日志做allreduce。
所有rank同时零token时抛错；单个rank零token时返回有图的0并参与通信。
这个API固定采用SUM/global_count约定，后续grad只SUM不平均。
普通DP/DDP平均梯度约定需要乘data_axis.size，不能不改归约就直接替换。


独立验收：`python -m megatron_mini.lab --case text`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T55 · Compose v2 training runtime

Day 7；位置 `training/engine_v2.py`；依赖：T21, T49。


`build_training_runtime`（生成时行号 31）：

T55: 将已验收模块装配为TrainRuntime，config.validate()后按recipe建图。
顺序: T21创建groups -> dense/llama/moe stage -> TP/SP/CP布局 ->
block/attention重算 -> sharding(ddp/zero1/zero2/fsdp/torch_fsdp2) -> optimizer。
再绑定GPipe/1F1B/VPP、mixed precision、GradRole及overlap hooks。
finish_communication完成全部角色归约后再检测overflow；finalize_and_clip只做
norm/clip，不再归约梯度。可复用T49，将第二次调用roles的reductions置为空。
旧T14只能用于纯dense旧布局，不得对FSDP/EP重复套average。
第一步先支持dense_tp_zero2；然后pipeline_1f1b；再moe_ep_fsdp。
不需要实现全部配置笛卡尔积，不支持的组合显式NotImplementedError并记录。
不能退回full dense单进程假装完成分布式。见docs/INTEGRATION.md逐行接线步骤。


集成验收：按 `docs/INTEGRATION.md` 对应接线步骤执行，不以默认pytest通过代替本任务完成。


## T56 · TP chunked communication overlap

Day 4；位置 `runtime/overlap.py`；依赖：无其他学生任务。


`tp_chunked_overlap`（生成时行号 40）：

T56: AG(chunk i+1) 与 GEMM(chunk i) 双buffer流水线，返回按序outputs。
launch_gather(i)->Pending，其buffers[0]为通信输出；compute(i,tensor)。
首次必须等第0块；消费前wait；不可在不同rank上动态改变chunk数量/顺序。
此为独立原语实验；用于SP时还须补相应RS backward，不能用无autograd拷贝。


独立验收：`python -m megatron_mini.lab --case overlap`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T57 · EP dispatch/compute overlap

Day 6；位置 `runtime/overlap.py`；依赖：T44。


`ep_chunked_overlap`（生成时行号 48）：

T57: dispatch(i+1)与expert_compute(i)重叠，再combine(i)。
dispatch返回Pending buffers[0]；compute(i,tensor)；combine(i,y)->output。
专家不同 token counts/空chunk也必须参与一致通信；返回按source chunk顺序结果。
可先双microbatch；不要依赖未知的下一个router结果提前发错all-to-all。


独立验收：`python -m megatron_mini.lab --case overlap`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T58 · CP ring double-buffer overlap

Day 3；位置 `runtime/overlap.py`；依赖：T29。


`cp_ring_overlap`（生成时行号 56）：

T58: exchange(i)返回下一KV buffer的Pending；consume(i)处理当前KV。
每轮先发下一块交换再算当前块，之后wait，调用者在exchange闭包里swap。
rounds对应逻辑KV块数；最后一轮无需再传已消费的KV。输出由consume闭包累积。
这里只验发送/消费顺序；T29负责online softmax与backward。


独立验收：`python -m megatron_mini.lab --case overlap`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。


## T59 · Typed pipeline transport adapter

Day 4；位置 `pipeline_parallel/transport.py`；依赖：T31。


`build_pipeline_io`（生成时行号 26）：

T59: 用TypedP2P和PipelineIO适配实际stage chunks。
首logical stage提供tokens；末stage提供loss；其余receive叶子activation。
forward peer按logical stage跨chunk wrap-around，backward反向；不把group rank
当global rank传给isend。BF16与FP32 payload遵从activation_spec。
为每step建立独立buffer队列；发送完成前不释放；最后统一finish。
验收pp2 bf16、pp2 vpp2与FP32 dense step对照；CPU仅验证FP32。


集成验收：按 `docs/INTEGRATION.md` 对应接线步骤执行，不以默认pytest通过代替本任务完成。


## T60 · Offline flat checkpoint reshard

Day 7；位置 `runtime/checkpoint.py`；依赖：T35。


`reshard_flat_state`（生成时行号 25）：

T60: 输入按old shard rank排序的一维tensor列表；剔除old padding，
按新world重新padding/切分；返回新的tensor列表，不修改输入。
分别用于FP32 master param、Adam m/v；step标量保持相同。
本任务仅flat shard的2->1/4离线重分片，不等价于任意TP/EP转换或在线弹性。
对总长度不足、负numel/非正world、dtype/device不一致明确报错。


独立验收：`python -m megatron_mini.lab --case reshard`。需要通信的case再加torchrun双进程；单进程通过不代表分布式通过。
