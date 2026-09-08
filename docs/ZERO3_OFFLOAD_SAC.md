# ZeRO-3、Offload、SAC：三个不同的内存优化维度

## 1. 模型状态分片，而不只是optimizer分片

| 策略 | 参数 | 梯度 | optimizer状态 |
|---|---|---|---|
| DDP | 复制 | 复制 | 复制 |
| ZeRO-1 | 复制 | 复制 | 分片 |
| ZeRO-2 | 复制 | 分片 | 分片 |
| ZeRO-3 | 分片 | 分片 | 分片 |
| FSDP full-shard路径 | 分片 | 分片 | 分片 |

表格描述持久状态；当前unit计算时仍可能有完整权重和临时完整梯度。mini先以数值正确的flat unit为界，不声称临时梯度峰值也完全消失。

ZeRO-3是算法分片阶段，DeepSpeed的ZeRO-3和PyTorch FSDP2不是同一个实现。v3用显式`zero3`入口提供逐unit封装和验收，底层复用手写FSDP的flat布局与AG/RS。PyTorch FSDP2用per-parameter DTensor表征，官方示例继续单列；不能套两层后声称显存除以DP两次。依据见[SOURCES_V3](SOURCES_V3.md)中的DeepSpeed与FSDP2官方文档。

## 2. Offload改变的是位置

参数分片决定“谁拥有”；offload决定“它放在哪”。CPU也只能保存本rank拥有的shard，不能所有rank都复制全模型而宣称主机内存缩放。

```text
CPU canonical parameter shard + FP32 master + Adam m/v
    ↓ H2D 当前unit的本地shard（可以只是同一FP32 owner的一份计算副本）
GPU local shard staging
    ↓ owner group parameter AllGather
GPU full parameter for current unit
    ↓ compute
release GPU full + staging after consumers finish
    ↓ backward时重新H2D + AG
local full-unit gradient
    ↓ ReduceScatter / 已定义的loss divisor
GPU local reduced gradient shard
    ↓ D2H + host wait
CPU最终梯度：累积、unscale、global finite/norm协调、clip
    ↓ CPU AdamW
更新CPU canonical shard，下次按需fetch
```

`param_gpu=off`和`optimizer_cpu=on`可以做optimizer-only路径；`param_cpu=on`则本mini要求optimizer也在CPU，以固定唯一owner、避免一周内扩展多种master-placement策略。这是课程约束，不是所有框架的普遍限制。

Activation offload是另一条路径：前向把选中的saved activations存入CPU，反向取回。它不是参数offload，也不是重计算。主线明确包括三类对象，不只有optimizer offload。

## 3. 同步优先，overlap后置验收

H2D之前需要生产者完成，消费GPU tensor之前需要copy stream完成；CPU使用D2H结果之前需要host-visible完成。`wait_event`向stream排队不等于CPU可以立即读host buffer。pinned memory加non_blocking也不保证所有路径都自动重叠。

最小实现允许先同步，之后复用T41/T65扩展prefetch和pool：记录一个buffer是FREE、COPYING、READY、IN_USE还是RETIRING，只有copy及consumer都完成才能复用。不能仅删除Python变量名或依赖empty_cache来实现正确回收。

## 4. SAC与全量AC

T33的block/attention checkpoint是**选择模块范围**，范围内通常重新执行所需计算。T67/T68的SAC是**范围内按算子决定保存还是重算**。在MLP基准中，保存mm/addmm等输出，重算便宜逐点算子。对真实attention必须重新看输出尺寸和SDPA实际op；节省的计算不应以保存巨大的score tensor为代价。

每个microbatch的SAC cache独立。forward保存的是当前权重下的结果，跨optimizer step复用会产生stale激活。存在dropout时保存/恢复RNG；replay后后续代码应看到与未重算路径一致的RNG推进。

### 不要用一次policy调用冒充一次实际计算

policy会在forward和recompute被询问；recompute时即使返回缓存，policy依然可能被调用。所以验收使用cache mode下面的ATen执行计数器：forward看到addmm，recompute不再实际执行对应addmm，但反向求梯度自己的matmul依旧需要执行。

## 5. 本课FSDP默认重算，必须专门适配SAC

v2 `_ShardedUnit` forward在no_grad下完成unit计算，backward全unit重算。直接给它外面包checkpoint会引入额外重算，里面包非reentrant checkpoint又不会自动建立预期的SAC cache。

因此v3增加 `_SelectiveUnit`：forward显式进入cache context，backward重新聚合参数后进入匹配的replay context。只缓存纯计算结果，AG/RS、prefetch和CPU copy不进入SAC。这样保留参数按需聚合语义，同时允许跳过某些昂贵前向算子的重算。

注意SAC cache不能保存完整参数的view/alias，否则full参数release失效。本课限制缓存是新计算产生的激活。整个GPU峰值中仍要计入SAC缓存。

### SAC + activation offload并不是两个context随便嵌套

saved_tensors_hooks作用于autograd保存的tensor，SAC内部的Python操作输出缓存不自动走该hook。首版组合明确只offload checkpoint边界输入/符合hook范围的saved activations；SAC保存的算子输出仍驻设备。如果需要把这些缓存也卸载，须实现另外的cache-store/backend，不可声称已经完成。

## 6. 显存和性能验收

从同一shape/global batch开始，对比DDP、ZeRO2、ZeRO3、ZeRO3+CPU optimizer、ZeRO3+CPU parameter、SAC、activation offload及指定组合。测量分为数值与资源两组：

数值组比较输出、输入梯度、每个本地参数梯度、3次更新和moments；overflow、裁剪、checkpoint resume另验。内存组不能同进程常驻dense oracle，否则oracle本身就占显存。使用独立进程/不同run，warmup包含Adam state初始化；同步并reset_peak_memory_stats后测steady-state peak，报告allocated/reserved、CPU RSS、pinned pool容量、H2D/D2H字节数及step time。

理想模型状态可粗略写成`(参数+梯度+optimizer)/shard_degree`，但完整unit、重算或SAC激活、预取unit、通信buffer和allocator余量都要另加。不能仅依据这个式子承诺显存减少倍数。

CPU/Gloo实验能检验数值和ownership；不能证明GPU内存下降、实际DMA、NCCL吞吐或overlap收益。

## 7. NVMe边界

主线交付GPU↔CPU，不实现NVMe。扩展时在CPU owner下面新增磁盘storage backend，必须补分块I/O、文件布局、校验、恢复、I/O completion、host-cache eviction、prefetch和写回一致性。把tensor torch.save到临时文件再load回来不等价于ZeRO-Infinity。配置当前拒绝`nvme`，不会静默改用CPU。
