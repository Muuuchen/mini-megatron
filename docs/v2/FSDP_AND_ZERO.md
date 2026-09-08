# 手写 FSDP 与 ZeRO：不是多加一个API开关

## 状态被谁拥有

| 策略 | 持久参数 | 持久梯度 | optimizer moments | 主要通信 |
|---|---|---|---|---|
| DDP | 每DP副本完整 | 每DP副本完整 | 每DP副本完整 | grad all-reduce |
| ZeRO-1 | 复制 | 复制 | shard | grad all-reduce + updated-param AG |
| ZeRO-2 | 复制 | shard目标 | shard | grad RS + updated-param AG |
| FSDP / ZeRO-3类 | shard | shard | shard | unit param AG + grad RS |

本课ZeRO-2逐参数grad→flatten的初始版本仍有完整梯度暂存。要体现内存收益，需把T34 bucket hook接进去、及时释放已归约buffer。
FSDP的full parameters是在计算时临时恢复的；如果始终保存full model再“额外”切一份shard，不能算参数内存分片。

## 手写单元的具体设计

`FlatLayout`记录name/shape/offset，允许参数跨shard边界，padding单独管理。
`FullShardUnit`只有`local_shard`注册为实际Parameter，template在meta设备上描述结构，没有持久full weight storage。
optimizer必须在wrap之后构造，操作local shard，不是保存旧完整parameter引用。

```
forward:
  local FP32 shard --AG--> full unit params
  functional_call(template, parameter_views, x)
  save only x / local shard / RNG / metadata
  release full params

backward:
  local shard --AG--> full unit params
  restore RNG in forked context; recompute unit under enable_grad
  autograd.grad -> dx + d_full_params
  CP sums if configured -> RS / explicit grad_divisor -> d_local_shard
  release full params and transient graph

optimizer:
  update local FP32 shard and local Adam m/v
```

这是一种**把单元重算与参数分片合并**的教学实现。它用额外计算简化生产FSDP的saved-tensor/view/hook管理。
因此不能拿其吞吐当作官方FSDP2的性能，也不能描述成“逐行复刻FSDP2”。真正要学的是ownership、materialization、释放点、梯度返回路径。
整数token输入不求dx，但embedding参数仍可有梯度。浮点activation允许传回dx；不得无条件对int64设置requires_grad。
RNG必须恢复以支持dropout重算，不得污染后续microbatch的RNG序列。

## 与TP、EP结合

先沿TP切GEMM参数，再在相应DP组上shard这些局部参数。AG后恢复的是本TP rank拥有的模块，不是其他TP shard。
expert参数只能在EDP组上做FSDP；dense/router参数用denseDP组。EP2+EDP2的最小实验需要denseDP4。
FSDP的RS如果已平均，T14/T49不再平均一遍。专家local mean梯度的除数不能误用EDP而漏掉EP贡献。

## 官方FSDP2对照

`fsdp/reference.py`已经提供bottom-up `fully_shard`适配器，`examples/torch_fsdp2_reference.py`是独立可启动CUDA实验。
FSDP2使用per-parameter DTensor sharding，和本课flat buffer不同；共同点是计算前AG、计算后释放及grad RS。
模块粒度、root是否reshard、prefetch策略会改变内存/通信权衡。不能只对root wrap后声称验证了逐层释放。

## 必需验收

先运行双进程`--case fsdp`，比较forward、local shard gradient、3次AdamW参数与m/v。
再做T40：至少两个units，检查每层都只持久保存shard，并测试多microbatch。
最后做T41：下一unit AG与当前unit计算重叠，记录峰值full buffers是否有界。
GPU上比较peak_allocated和step time；CPU/Gloo的RS fallback是SUM+slice，只能证明数值，不证明通信量或显存收益。

来源和版本说明见SOURCES_V2.md。官方FSDP2调用不能代替T38/T39的完成判据。
