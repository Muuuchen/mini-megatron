# T61–T72：新增12项实现契约

旧任务不改编号。逐项替换函数体中的 `todo('Txx')`，保留中文契约。任务移除占位符只表示“待验收”。下面命令均从仓库根目录执行。

## T61 · 参数驻留状态

`memory/residency.py::ShardHandle.materialize`：本地 shard 经当前 owner group AllGather 生成 full unit。严格执行 SHARDED→FULL→SHARDED，禁止在 full 尚未释放时重复聚合。记录调用数与 transient full bytes；`release` 已提供。

验收：`lab --case residency`。代码计数器不能发现所有外部 storage alias，GPU还要检查峰值、ctx saved tensors 和实际引用链。

## T62 · 显式 ZeRO-3 入口

`optimizer/zero3.py::build_zero3_sequential`：保留 immediate child 名字和顺序，逐 child 创建 FullShardUnit；`sac=True` 时创建 SACFullShardUnit。拒绝跨 child tied storage，不能留一份原始完整模型作为成员。optimizer 在 wrap 后创建。学习对象是参数/梯度/optimizer 全分片，不是再复制T38/T39。

验收依赖 T35/T38/T39：`torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case zero3`。测试两个unit、两个同时存活的microbatch图、连续3步、输入梯度、local param grad、全局范数裁剪和Adam m/v。AG/RS次数也检查。

## T63 · CPU shard optimizer

`offload/optimizer.py::CPUShardAdam.step`：输入已经归约、平均、unscale、裁剪的最终局部梯度；调用T65搬到CPU，host wait 后写CPU master.grad并更新一次CPU AdamW。master/m/v不得跑回GPU，CPU也不能每rank保存全模型optimizer。

`found_inf=True` 表示上游已经跨所有参与rank一致协商，master/m/v/step不变。独立case输入是已归约梯度，不负责再次同步。验收：`lab --case offload_optimizer`，含连续3步、overflow skip、恢复后下一步。

## T64 · 参数CPU offload

`offload/parameters.py::CPUParamStore.materialize`：T65 H2D local shard，消费stream等copy完成，再参数AG；返回当前unit完整计算权重。CPU参数必须与optimizer.master共享唯一owner。禁止CPU tensor进NCCL，禁止常驻完整GPU模型，也不能每次从未更新的旧CPU副本取值。

验收：双进程 `--case offload_param`。第一次materialize、release、CPU更新、第二次materialize必须反映新参数；检查staging/full引用释放。GPU加 `--device cuda`；CPU只做placement emulation。

## T65 · DMA顺序与lease

`offload/transfer.py::submit_copy`：申请独立snapshot、处理非连续输入、保持source/destination直到copy完成。CPU path同步；CUDA path明确producer stream→copy stream→consumer stream依赖，记录event。

`Transfer.wait_stream` 只建立GPU顺序；CPU访问D2H结果必须 `wait_host`。即使保留Python引用，CPU提前写入pinned源仍会发生数据竞争；lease的使用约束必须落实到buffer pool。

验收：`lab --case offload_transfer`，检查快照不是alias、双向复制和host可见性。真实异步行为只用CUDA确认，CPU mock不构成overlap证据。

## T66 · 激活pack/unpack

`offload/activations.py::ActivationOffloader.pack`：通过saved_tensors_hooks保存host snapshot，原设备/shape/dtype一起记录；unpack/drain框架已给。保护参数的storage（包括转置view），不要依据 `is_leaf` 错把参数当激活。

普通CPU输入不offload，除非显式 `emulate_cpu=True`。int tensor不offload。本版不支持in-place修改被保存的tensor或依赖alias的计算。框架在forward结束时drain D2H以回收GPU源引用，这是同步正确性基准，不宣称异步收益。

验收：`lab --case offload_activation`，比较输出/输入梯度/参数梯度，验证非连续snapshot、参数view保护和host-only记录。**saved_tensors_hooks并不捕获任意Python缓存中的tensor，尤其不能自动offload SAC op-cache。**

## T67 · SAC policy

`runtime/sac.py::save_matmuls_policy`：精确匹配ATen OpOverload；对mm/addmm/bmm返回MUST_SAVE，其他纯局部op返回PREFER_RECOMPUTE。这是MLP基准policy，不是通用最优Transformer policy。

不要缓存参数切片/view；不要把通信/optimizer写操作/状态化token dispatch放进重算区。SDPA后端不同，实际op不同；保存S×S attention分数可能适得其反。参数之外的缓存必须计入显存。

## T68 · SAC执行入口

`runtime/sac.py::selective_checkpoint`：使用非reentrant checkpoint和每次调用独立的paired contexts，保留RNG。`observed_contexts`已提供，它将计数器放在cache mode下方，看到的是实际执行的算子，不是“被策略询问过”的算子。

验收：`lab --case sac`，含两个microbatch先forward后反向、dropout/RNG、梯度和重算区内addmm实际调用为0。默认完整checkpoint是T33，与这里分开。

## T69 / T70 · full-shard SAC forward/backward

`fsdp/selective.py::_SelectiveUnit.forward/backward`：每次forward创建并保存独立SAC缓存上下文。参数AG在上下文之外；只在纯函数unit.call周围进入cache-forward和cached-replay。ctx保存x/local_shard/RNG/metadata/cache，不保存完整flat参数。backward重新AG恢复当前unit参数，enable_grad构造本地图；被缓存的GEMM输出可以复用，但求梯度的GEMM仍要正常执行。

不能把普通 `checkpoint` 放进旧 `_ShardedUnit.forward` 的no_grad区域就结束：普通非reentrant入口在这种情况下不会建立所需的SAC执行过程。也不能在原来的强制全unit重算外再套全block checkpoint，导致多次AG和重复计算。

验收：`torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case zero3_sac`。与zero3同样的参数/梯度/裁剪/3步/moments对照，并检查SAC实际matmul执行数。多个live graphs不能共享同一缓存实例。

## T71 · memory组合接线

`memory/integration.py::build_memory_runtime`：由T55调度，返回TrainRuntime，支持指定的zero3_cpu、zero3_cpu_sac、fsdp_cpu_sac。统一CPU canonical shard、GPU staging、SAC replay、grad accumulation、全局overflow、裁剪和CPU update。

为了复用本课FullShardUnit，可以将unit.local_shard设为CPUShardAdam.master，CPUParamStore引用同一Parameter；抽出unit.materialize/release/finish_gradient接口，替换T38/T39或T69/T70中直接AG的部分。CPU shard作为Function输入时，backward必须返回CPU dlocal。保留no-offload路径让已有case继续通过。

先集成FP32、DP2、两unit、M2；再BF16/FP16与并行组合。grad已经RS之后不再套普通DP平均。GPU collective的overflow/norm需要device flags或标量，不能拿CPUgrad tensor直接调用NCCL。整步必须统一skip/update。

**验收是一个独立的最终实验，不被上述单模块case覆盖。** 至少对照3次更新、裁剪开关、一个rank注入inf、CPU/GPUowner同步、恢复下一次更新。启用activation offload时，先显式卸载保存的checkpoint输入；SAC内部缓存仍在设备，除非另写并验证缓存存储后端。

## T72 · offload快照与恢复

`offload/checkpoint.py::snapshot_offloaded`：等待transfers，复制CPUowner state，带format_version/next_step/metadata。真实文件提交复用T50的manifest；不保存SACcache、完整临时参数或旧CUDA事件。

验收：`lab --case offload_checkpoint`，检查等待顺序、快照与live tensor无storage alias、load后下一步相同。global barrier/文件事务仍由T50集成，不由本函数单独承诺。
