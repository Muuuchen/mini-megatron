# 七天密集实践计划 · v2

所有列出的机制都是本版必修，不再把 MoE/FSDP/SP/CP/overlap 写成“可选扩展”。
工作量按已有 autograd/CUDA/分布式基础估算约45–60小时；一周是学习目标，不是对所有实现与性能验证的保证。
这七天是**覆盖机制 + 完成指定组合**，不是复制整个 Megatron 工程。

## Day 1 · 基础训练、rank 语义、模型组件

任务：T01–T07、T10–T13、T15、T21、T51–T54。
先跑 baseline，基础结构题按已有 dense oracle 填，不花时间重造训练器。
读 TP GEMM 的输入/输出分片，写出 `WORLD=TP*PP*CP*DP` 和 EP/EDP 子组。
RMSNorm/RoPE/GQA 做单元公式题；真实文本仅 UTF-8 byte token，首先明确 mask 分母。

```bash
python -m pytest -q
python -m megatron_mini.train --mode baseline --steps 3
python -m megatron_mini.lab --case topology
python -m megatron_mini.lab --case model
python -m megatron_mini.lab --case text
python -m pytest -q --exercises tests/test_exercises.py -k 'student_model or student_gradient'
```

验收：完整 GPT 单进程 forward/backward 对齐；解释同一模型副本共享数据，DP分片消费不同样本。

## Day 2 · TP、词表 CE 与 SP

任务：T08、T09、T14、T22–T25。
先 vocab CE，再 SP。SP 不能直接复用旧 column 的 copy-backward，否则 dX 重复SUM。

```bash
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case loss --tp 2
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case step --tp 2
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case step --dp 2
torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case sp
```

验收：词表 target 跨 shard、nonzero row bias、SP norm/bias 的 TP SUM。loss下降不能替代梯度对齐。

## Day 3 · CP、重计算、精度与 CP overlap

任务：T26–T29、T33、T47、T48、T58。
先 CP gather 正确性基准，明确 Q 的全局位置；然后 online softmax、ring forward/backward。
重算使用非reentrant checkpoint；精度采用FP32 master，BF16不强加scaler，FP16做统一overflow。

```bash
torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case cp
python -m pytest -q --exercises tests/test_v2_exercises.py -k online_update
torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case ring
python -m megatron_mini.lab --case recompute
torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case precision
```

验收：全masked KV块不NaN，CP2 的 dK/dV回owner；dropout重算不改变RNG；一个rank注入inf后所有rank跳过step。
Ring是本日难点，CPU验证不代表显存或吞吐收益。

## Day 4 · PP/1F1B/VPP 与 overlap primitives

任务：T16、T17、T30–T32、T34、T56、T59。
保留GPipe比较基准，先写event plan，再接已提供的typed异步P2P。
VPP=2将logical stage映射为c*PP+p，处理最后physical stage到第一个的wrap-around。

```bash
python -m pytest -q --exercises tests/test_v2_exercises.py -k '1f1b_events or interleaved_events'
torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case pipeline
torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case interleaved
python -m megatron_mini.lab --case bucket
```

验收：每个(microbatch,chunk)恰好一次F/B，M<PP不死锁，buffer发出后不能提前复用。
DP bucket、TP chunk的时序测试与GPU overlap实验要分开记录。

## Day 5 · ZeRO 与手写 FSDP

任务：T18、T35–T41。
共用FlatLayout，先ZeRO-1→ZeRO-2，再手写FSDP unit生命周期。
**手写版需要在forward结束释放full参数，backward重新聚合并重算。**
官方FSDP2对照不能算T38/T39完成；说明两种实现的不同内存/算力代价。

```bash
python -m megatron_mini.lab --case flat
for c in zero1 zero2 fsdp; do
  torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case "$c"
done
# 仅有两张可见GPU、兼容PyTorch时执行；这份reference不依赖学生TODO。
torchrun --standalone --nproc_per_node=2 -m examples.torch_fsdp2_reference
```

验收：连续3次AdamW参数和m/v对齐；跨参数边界分片；unit内只有local_shard是真实持久参数。
ZeRO-2若仅step末flatten，必须记为语义版本，而非完整gradient-memory优化。

## Day 6 · MoE、EP/EDP、EP overlap

任务：T42–T46、T57。
top2→permutation→counts exchange→variable AllToAll→local experts→reverse AllToAll→weighted combine。
必须区分router/dense参数和expert参数的归约组，覆盖空专家、0-size split、偏斜路由。

```bash
torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case moe
python -m megatron_mini.lab --case overlap
```

验收：router有主loss/aux梯度，多个expert结果相加而非覆盖；expert grads只在EDP同步。
`--case moe`基准为EP2/EDP1；EP2/EDP2需Day7四进程组合，不可混淆。

## Day 7 · 组合训练、正确归一化、恢复

任务：T19、T20、T49、T50、T55、T60。
按INTEGRATION.md逐个接线，先dense TP×ZeRO2，再PP 1F1B，再MoE EP×FSDP。
每个新增组合保持此前独立case仍然通过。所有梯度归约结束后再unscale/check-inf，再clip/step。

```bash
python -m megatron_mini.lab --case roles
python -m megatron_mini.lab --case checkpoint
python -m megatron_mini.lab --case reshard
python -m megatron_mini.lab --recipe dense_tp_zero2
# T55及依赖补完后才允许实际启动：
torchrun --standalone --nproc_per_node=4 -m megatron_mini.lab --recipe dense_tp_zero2 --run
torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --recipe pipeline_1f1b --run
torchrun --standalone --nproc_per_node=4 -m megatron_mini.lab --recipe moe_ep_fsdp --run
```

课程结项必须提交“通过了哪些组合、在哪种设备、比较了什么”。集成入口的loss打印并非完整验收；
需按INTEGRATION.md比较梯度、3次更新和恢复后下一步，记录到ACCEPTANCE_MATRIX.md。
没有GPU时可以完成多数语义实验，但CUDA/NCCL与真实overlap一栏必须保留未测。
