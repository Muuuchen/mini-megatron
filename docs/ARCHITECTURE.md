# 架构、布局与梯度归属

## 两条路径不是同一个成熟度

v1 `train.py -> training/engine.py`保留单卡baseline及TP/DP/GPipe教学任务，未改成满是特性开关的大函数。
v2 `lab.py -> labs/checks.py`直接验证独立机制；`lab.py --recipe ... --run -> build_training_runtime(T55)`负责组合。
框架给出配置、原语、模块构造、事件类型、生命周期、reference和失败预检；算法与最终接线由学生完成。

## 并行坐标

坐标 `[d,p,c,t]`，TP最快变化；`rank=((d*PP+p)*CP+c)*TP+t`。
本课不实现 Megatron 的通用 parallel folding，采用一个明确的EP-on-DP布局：

```
WORLD = TP * PP * CP * denseDP
EP | denseDP
expert_DP = denseDP / EP
SP复用TP组；FSDP复用适当的数据复制组；VPP不是额外物理rank维度。
```

T21必须按统一顺序创建tp/pp/cp/dp/ep/edp/dp_cp/edp_cp，所有进程都枚举全部组。
EP固定d//EP、变化d%EP；EDP固定d%EP、变化d//EP。
Dense参数在各DP rank复制；expert参数在不同EP rank上不同，不能沿EP做参数梯度平均。

## 激活布局

常规TP残差 `[B,S,H]`复制，column输出 `[B,S,F/TP]`，row回到完整H。
SP残差和norm为 `[B,S/TP,H]`；column前AG的反向是RS，row后的RS反向是AG。
SP norm/row bias只看到自己的tokens，因此需要额外TP SUM梯度；旧TP的copy mapping不得重复叠加。

CP输入/Q/输出按S分片，localQ需要globalKV。gather baseline只聚KV；ring保持块级KV并在线合并softmax。
SP+CP时residual可为 `[B,S/(CP*TP),H]`：CP先定义token区间，SP在该区间内切分；positions始终全局。
构造时检查S能被所用sequence维度整除，不允许每rank私自padding改变分母。

## 两种合法loss约定，不得混用

独立ZeRO/FSDP/MoE实验使用每rank等长local mean。Dense梯度SUM后除denseDP×CP；SP norm另外沿TP求和。
EP dispatcher已把不同EP源token贡献送到expert owner，expert在EDP×CP上SUM，但分母仍为denseDP×CP。
FSDP unit的`grad_divisor`显式配置；`grad_sum_axes`可承接CP贡献，避免在engine重复平均。

变长文本T54采用另一约定：local有效loss sum/global有效token数，梯度后续全部SUM。
它不是直接把原来的mean CE替换掉就结束；必须连同步除数一起改变。
Aux loss也要明确统计范围与缩放，不能每个TP副本再次累加。

## 参数所有者与范数

T49的GradRole不是装饰字段：每个参数记录哪些轴上有不同梯度贡献、除数、是否计入范数、是否sharded。
DP复制参数norm只计一次；TP shards全部计，TP replicated参数只取一份；专家按expert ID只计一份；
FSDP local shards全部计入但padding排除。逻辑范数不等于直接对WORLD上所有物理tensor做SUM。
计数去重和更新行为不同：所有需要同步的副本仍要被同样裁剪和更新。

## 训练生命周期

```
zero_grad
  -> local microbatch forward/backward（scale在本窗口内固定）
  -> finish all role reductions / bucket / RS / P2P
  -> unscale once + global found_inf agreement
  -> logical norm / clip
  -> local optimizer shard step
  -> parameter materialization或下一轮prefetch
  -> checkpoint（没有未完成的work）
```

TrainRuntime的finish_communication必须完成所有梯度归约，不只wait已发出的bucket；否则finite check可能太早。
finalize_and_clip不得重复做已完成的归约。T49可拆成内部helper或第二次调用空reductions的roles。

## 为什么不能开async_op就宣布通信重叠

Async只改变提交/等待接口。真正重叠还要求独立计算、正确的stream依赖和活跃buffer生命周期。
DP等bucket ready；FSDP提前聚合下一unit；TP按chunk；PP持有发送buffer；EP必须先有路由metadata；CP双buffer环。
FakeWork实验只验证依赖和顺序；CUDA profiler trace与端到端step time才判断有没有收益。
