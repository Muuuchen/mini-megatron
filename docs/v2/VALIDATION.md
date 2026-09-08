# v2 实际验证记录

记录日期：2026-09-08（Asia/Singapore）。环境：Python 3.13.5、PyTorch 2.10.0+cpu，CUDA不可用。

## 原样交付 starter

```
python -m pytest -q
37 passed, 31 skipped
```

37项通过的是已实现脚手架、配置、rank arithmetic、参考模型、生命周期、registry、event validator等。
31项跳过是显式`--exercises`才启用的学生测试，包含v1的7项与v2的24项。
**60个学生TODO仍在。默认pytest通过不代表TP/SP/CP/PP/EP/ZeRO/FSDP已经实现。**

已执行：compileall；dense baseline CPU 3 steps；task/case inventory；recipe规划输出；
`lab --case fsdp`在创建任何通信组前明确报出T35/T38/T39；真实2进程CPU/Gloo的Axis SUM/AG/RS适配和typed P2P。
日志见`validation_logs/`。
默认build isolation安装因沙盒无法解析依赖源失败；使用已有setuptools的
`pip install -e . --no-deps --no-build-isolation`本地安装成功，网络失败与成功日志均保留。

## 临时填充副本：只验证框架能衔接

单独复制源码到临时目录，用独立验证实现填入部分TODO。临时答案**不在交付ZIP内**；
这些测试不能被当作学生完成记录，也不能把结论外推到未测试模块。

| 实际测试 | CPU进程 | 比较范围 | 结果 |
|---|---:|---|---|
| SP MLP + LayerNorm | 2 | forward、local dX、sharded dW、replicated norm/bias grad | PASS |
| CP gather attention | 2 | global causal mask、forward、dQ/dK/dV | PASS |
| ZeRO-1 | 2 | 连续3次AdamW参数、m/v | PASS |
| ZeRO-2语义路径 | 2 | 连续3次AdamW参数、m/v | PASS |
| 手写FSDP单unit | 2 | forward、shard grad、连续3次AdamW参数及m/v、meta template | PASS |
| MoE top2 EP2/EDP1 | 2 | forward、router/expert grads、aux可微 | PASS |
| 非交错1F1B | 2 | M=4、多stage参数梯度对照完整MLP链 | PASS |

日志文件名含`temp_filled`，和starter transport日志严格区分。
ZeRO-2用的是step末full-gradient buffer数值路径，没有证明真实梯度内存节约。
Gloo的reduce-scatter适配使用SUM+slice，不能作为NCCL通信量、内存或性能证据。
MoE/1F1B这次没有执行多步optimizer更新；不能描述成对应三步训练已测。

## 尚未验证 / 待学生实现

Ring attention的真实环形backward；VPP/interleaved实际传输；T40多unit模型接线；T55组合训练后端；
T59完整GPT typed IO；全部角色的混合并行gradient/norm；完整checkpoint manifest/恢复/reshard；
BF16/FP16真实GPU训练；官方FSDP2 CUDA示例；所有真实CUDA stream overlap与性能。

FakeWork/event simulator证明的是局部顺序/依赖，不是跨rank完整通信安全性或GPU重叠收益。
新增recipe目前是**规划 + 强制TODO预检 + 已定义运行时接口**，不是已实现全部配置的自动训练器。

## 学生完成后的证据要求

独立case从1/2进程开始，再接指定4进程组合；每次记录源码状态、数据形状、dtype、tol。
组合训练应比较forward/grad/3次参数更新/optimizer moments以及resume下一步，不能只观察loss下降。
在`docs/ACCEPTANCE_MATRIX.md`填写实际结果，未测的硬件/组合保持未测。
