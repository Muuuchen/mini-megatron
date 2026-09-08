# 主线之外的扩展：先通过数值矩阵，再选一个方向

以下全部未实现，不计入一周的 20 项任务。扩展不是加一个开关就宣布支持，需要新增对应语义测试。每次只引入一个变量，保持当前 dense oracle 可用。

## X01 · 非交错 1F1B（最推荐的下一步）

在 `pipeline_parallel/` 新增独立 scheduler，不直接覆盖已经可用的 GPipe baseline。先实现纯 Python 的 event plan，区分 warmup、steady state、cooldown，再接 P2P。

需要显式维护 microbatch ID、待反向图队列、发送/接收伙伴。steady state 若两个相邻 stage 都先阻塞 send，就可能形成环形等待；不能把当前 GPipe 的 send 顺序直接机械搬过来。应设计正确的 send/recv 配对或组合通信，并检查非阻塞 buffer 生命周期。

验收从 PP2/M1、PP2/M4、PP4/M1、PP4/M8 开始，比较每个 microbatch 恰好一次 F/B、参数更新与 GPipe 等价，再测 peak live activation records。只有测到更低峰值与正确吞吐，才讨论其优势。

## X02 · Activation recomputation

先在完整 block 边界使用 PyTorch checkpoint 机制，保持所有并行语义不变。主线无 dropout，可先排除随机状态问题；后续引入 dropout 后必须独立验证重算的 RNG 一致性。

验收：输出/梯度/更新等价，同时记录重算额外计算和峰值激活内存。不要把 parameter checkpoint 保存/恢复与 activation checkpointing 混为一谈。

## X03 · Sequence Parallelism

这不是让每个 TP rank 简单切一段 tokens 后继续原来的网络。需要重新标注残差/LayerNorm 的 sequence 布局，明确 TP GEMM 入口/出口的 all-gather/reduce-scatter 及相应 autograd 契约，并处理 replicated 参数梯度同步。

先只做 MLP+LayerNorm 的小图，比较前后向和内存，再扩到 attention/GPT。不能未经验证就把 gather 的 backward 一律写成 split；它取决于上游逻辑副本和梯度贡献是否需要求和。

## X04 · DP buckets 与通信重叠

保持 T14 的串行逐参数平均作为 correctness baseline。新增 bucket 布局、ready 计数和异步 work handles，保证某个 bucket 的最后一次 microbatch 梯度就绪后才启动同步，optimizer.step 前全部 wait。

验收：小/大 bucket、不同参数顺序、microbatch>1 都与原先梯度一致。测 overlap 时区分 launch overhead、带宽、等待和总 step time，不仅看某一条通信 kernel 的时长。

## X05 · BF16

当前 PP payload 固定 FP32，所以需要同时设计参数 dtype、activation dtype、通信 buffer dtype、loss accumulator、optimizer state，而不是只对 stage 调 `.bfloat16()`。

先单进程，再 TP，再 PP，再混合。数值验收用合理 tolerance 和多步稳定性，FP32 对齐测试继续保留。FP16 的 loss scaling 和 FP8 的 scale/amax 管理不在此扩展的最小范围内。

## X06 · 真实文本、RoPE/RMSNorm/GQA

保持分布式核心不变，逐个替换模型构件。先写新的 dense oracle，再改 parallel model。GQA 需要明确 Q heads 与 KV heads 的各自分片/复制约束；不要假设 Q heads 能均分就代表 KV heads 也能。

真实文本首先做固定长度、无 padding 的拼接样本；随后再设计有效 token count 和 mask 的全局归约。未解决分母问题前，不要把当前 loss/M 直接用于变长 batch。

## 暂不建议紧接着加入

MoE/EP、CP、ZeRO/分布式 optimizer、跨拓扑 checkpoint、容错训练都属于新的系统子课题。完成并行基础之后，按自己的研究目标选一个，而不是继续堆配置字段。
