# T55 集成接线契约

这是**保留给学生的集成任务**。已存在的recipe目录、CLI和TrainRuntime并不代表其后端已实现。
本文件给出装配顺序和验收要求，不通过一个最终`todo()`掩盖缺失实现。

## 第一步：只接一个组合

先用`dense_tp_zero2`（TP2、DP2、SP）。所有相关独立case先通过。
1. 从T21的全世界GroupSpec顺序调用new_group，保存当前rank各Axis及global peer ranks。
2. 用v1 GPTStage结构作为模板；在block里选择SP MLP/norm路径；embedding/head处显式转换sequence布局。
3. Attention里Q/K/V和输出投影对应同一SP边界模式，不可用旧copy+SP AG两次SUM dX。
4. 一个DP group里模型参数属于同一个TP/PP shard；初始化从相同dense state切片，然后创建ShardedOptimizer。
5. 绑定forward_backward_no_pipeline式microbatch循环、FP32 loss/M，step外才进行grad同步。
6. **ZeRO策略的step要拆成prepare_grad、local_update、publish_params阶段**：不能把现有ZeRO.step里已带的
   归约再由T49做一次；也不能在clip之后重新从未裁剪的旧grad复制一份覆盖掉local_shard.grad。
   可以抽出共用helper，保持`lab --case zero1/zero2`仍通过。
7. finish_communication完成SP/DP角色归约；unscale/check在这之后；finalize_and_clip只对实际将被optimizer消费的grad裁剪。
8. optimizer step后allgather updated parameter shards，下一microbatch前完成发布。

**验收不能只有loss打印**：相同seed/global batch对照dense，检查参数分片对应关系、梯度、3次AdamW更新、optimizer m/v。
对SP的norm/bias与TP分片权重分别检查；clip开启和关闭都要测。

## 第二步：pipeline_1f1b

接T30 event plan → T59 PipelineIO → T31 executor。每个stage只实例化本地layers。
输入embedding仅第一logical stage、final norm/head仅最后logical stage。最终loss/M只乘一次。
IO传输activation dtype必须与实际compute输出一致，payload与梯度buffer都要检查shape。
必须保留GPipe对照；先PP2/M1与PP2/M4，再T32的PP2/VPP2/M4。
VPP传输逻辑不能从non-interleaved的peer与microbatch规则盲目复制。

## 第三步：moe_ep_fsdp

recipe为denseDP4、EP2、EDP2。把FFN换成ExpertParallelMoE并设置dense/expert两类FSDP units。
router/dense units用denseDP4；local experts用EDP2；不同expert ID从不互做参数AG。
手写unit的`grad_sum_axes`负责CP等额外贡献（此recipe CP=1），`grad_divisor`负责整个loss约定。
EP AllToAll已经汇合源token梯度，因此expert除数与EDP degree并非总相同。
初始化同时准备一个全部专家的dense oracle，仅用于测试，不能留在每rank生产训练模型里。

验收先top2均衡，再偏斜路由和空专家。比较router grad、每个expert grad和3次更新；all_to_all backward不除EP。
恢复时同时保存expert全局ID、flat layout、m/v、scaler、RNG与data cursor。

## 后续接线也属于本课程范围

SP+CP：定义global token position和两级sequence切片；不在每个CP rank重置RoPE。
MoE+TP+EP：必须重新标注进入dispatcher的token是否已按SP去重，不能发送重复的TP token副本。
BF16/FP16：compute activation policy、P2P dtype、FP32 grad reduce、同一步overflow一致性同时切换。
Overlap：先用串行模式通过完全相同的数值测试，再接异步；T34/41/56/57/58各自只改变通信提交与消费顺序。

## TrainRuntime字段定义

`zero_grad()` 清本步梯度但不清optimizer moments。
`forward_backward(microbatches)` 执行已选调度；输入数据由runtime搬到设备，并按CP/SP规则切分。
`finish_communication()` 发起并完成所有必须的梯度归约，等待已有bucket/P2P；不要先clip。
`unscale_and_check()` 只unscale一次并在所有参与step的rank统一found_inf。
`finalize_and_clip()` 只算最终梯度norm并裁剪，不能再次平均。
`optimizer_step()` 更新正确的shards，并发布下一步要用的参数。
`update_scale(found_inf)` 每个完整microbatch窗口结束调用一次。

某个recipe尚未接好时显式报错并在验收矩阵填“待接线”，不能fallback到v1训练声称支持。
不要求七天完成任意参数组合的笛卡尔积；至少完成上面三个有价值组合，并把其他机制的独立实验验收做齐。


## v3 第四步：zero3_cpu_sac（T71）

此recipe需要DP2；初版先不同时叠加PP/EP/TP。T55负责模型/mesh构造，T71把CPU owner、unit staging、SAC replay和TrainRuntime字段接起来。禁止T55→T71→T55循环调用；抽出shared setup helpers。

CPU optimizer的master必须同时是unit的canonical local Parameter，CPUParamStore只能引用它。CPU输入Parameter跨自定义Function到CUDA计算时，backward返回CPU local grad；AG/RS在GPU staging上进行。microbatch累积后统一finite check和clip，更新时CPU owner/moments/step只能推进一次。

保存时先T72 drain并取得CPU-only snapshot，再T50全体rank提交。恢复重建staging/events，不能从旧SAC cache取激活。recipe验收需独立编写端到端对照并记录结果；`scripts/check_memory.sh`只验独立机制，不等于recipe已通过。

所有FP8之外的v2机制保留，但任意组合仍需显式验证；不支持的组合抛NotImplementedError。不要只添加recipe然后在train.py静默回退。
