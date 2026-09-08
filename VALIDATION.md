# v3 实际验证记录

日期：2026-09-08。环境：Python 3.13.5、PyTorch 2.10.0+cpu、pytest 9.0.2；`torch.cuda.is_available() == False`。

## 1. 原样交付的 starter

- `python -m pytest -q`：**50 passed, 40 skipped**。
- `python -m compileall -q megatron_mini tests examples`：通过。
- dense baseline CPU 训练3步：通过。只用于确认入口，CPU吞吐不是性能结论。
- `tasks`：**72/72 TODO保留**，旧T01–T60不改号。
- `--recipe zero3_cpu_sac`：配置和依赖可检查，不执行训练。
- `--case zero3` / `--case sac`：缺实现时在创建进程组前明确退出，未回退到reference。
- `observed_contexts` instrument测试：保存的mm在replay没有实际执行，sin仍执行；此项只验证已提供的计数器，而不是T67/T68已完成。

日志：`validation_logs/v3/starter_pytest.txt`、`baseline_cpu.jsonl`、`tasks.txt`、`*_preflight.txt`。

## 2. 不交付答案的临时填充副本

为确认新接口和验收可衔接，另建临时副本，只填充新case需要的CPU实现及T35/T38/T39依赖。交付包中的占位符没有被替换。

单进程：`pytest --exercises tests/test_v3_exercises.py`，**9 passed**。

真实两个进程，Gloo，各case分别启动并通过：

| case | 实际检查 |
|---|---|
| residency | 两rank gather、状态、重入拒绝、引用释放与记账 |
| zero3 | 两unit、M=2、同时存活的图、3次AdamW更新、dX/dW、clip、moments、AG/RS次数 |
| sac | dropout/RNG、双microbatch cache、梯度、实际执行计数 |
| zero3_sac | 分片与SAC联合的小模型数值检查，cached addmm不再执行 |
| offload_transfer | CPU复制模拟、非连续输入、snapshot非alias；无GPU DMA |
| offload_optimizer | CPU master/m/v/step、3次更新、overflow不推进、resume下一步 |
| offload_param | CPU模拟staging/AG、release、CPUowner更新可见 |
| offload_activation | CPU保存模拟、参数view保护、值/梯度和snapshot |
| offload_checkpoint | transfer flush、CPU snapshot隔离、恢复后下一次更新 |

日志：`validation_logs/v3/temporary_filled_local.txt` 和 `temporary_filled_cpu2/*.txt`。这些日志只说明**所设计的小规模CPU路径可以被实现并通过测试**，不代表starter已有算法，也不代表GPU路径可用。

## 3. 明确未验证

**T55/T71的完整`zero3_cpu_sac`/`fsdp_cpu_sac`训练recipe未接线、未验收。** 独立CPU offload case与zero3_sac case通过，不等于ZeRO3+CPUOffload+SAC三者全链路组合通过。

GPU H2D/D2H、pinned-memory异步lease、NCCL、CUDA多stream、GPU峰值显存、吞吐/通信重叠均未实测。官方FSDP2新增`--cpu-offload`只做语法/import检查，未做CUDA运行。BF16/FP16的新增memory组合未实测。

未实现NVMe。未新增验证v2未完成的ring backward、VPP transport、任意多维并行组合；这些历史限制不因版本号更新而消失。旧验证日志/旧文档保留作历史参考，不是本次新执行结果。

## 4. 数值与资源验收分开

CPU数值通过不能证明GPU内存下降。GPU内存实验需单独run，不能与dense oracle常驻同进程；warmup包括optimizer moments初始化，记录allocated/reserved、host/pinned容量、传输字节和最大rank step时间。SAC缓存也要计入，不声称泛用saved_tensors_hooks能自动offload SAC内部缓存。
