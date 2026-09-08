import torch
from megatron_mini.exercise import todo
from megatron_mini.distributed.mesh import sum_


@torch.no_grad()
def average_dp_gradients(model, mesh):
    """T14: 一个 optimizer step 中累积完全部 microbatches 后同步一次。

    所有参数在 dense GPT 中每步都参与计算，参数遍历顺序在同一 DP group 一致。
    grad=None 在本 lab 表示 bug，应该报错，不要静默跳过 collective。
    每个 grad 做 DP SUM 后除 dp_size；不能对 TP group 或 WORLD 求平均。
    这里不用 DDP hooks/bucket/overlap，方便先理解正确语义。
    """
    if mesh.dims.dp == 1:
        return
    todo("T14")


@torch.no_grad()
def clip_grad_norm(model, mesh, max_norm: float) -> torch.Tensor:
    """T18: 对一个逻辑完整模型计算 L2 grad norm，并统一裁剪所有本地梯度。

    在 DP average 之后执行。计入 shard 参数的全部本地平方和；replicated
    参数只在 tp_rank=0 计入。然后先 TP SUM，再 PP SUM；不在 DP 上求和。
    注意是“计数去重”，不是让其他 TP rank 不裁剪自己的 replicated 参数。
    返回裁剪前 norm 标量；所有 rank 用相同 factor=min(1,max_norm/(norm+1e-6))。
    max_norm <= 0 为参数错误。使用 fp64 累加平方和增强测试稳定性。
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    todo("T18")
