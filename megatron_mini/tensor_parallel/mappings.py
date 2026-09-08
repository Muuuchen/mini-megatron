"""TP 逻辑张量的自动微分契约，不是通用 all_reduce 的梯度定义。"""
import torch
from megatron_mini.exercise import todo
from megatron_mini.distributed.mesh import sum_


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mesh):
        ctx.mesh = mesh
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        """T03: forward identity；backward 对 TP 分支的 dX 求 SUM。

        返回 (grad_x, None)；SUM，不是 mean。先 clone contiguous，不修改传入 grad。
        mesh.size('tp')==1 时也应正确。
        """
        todo("T03")


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, partial, mesh):
        """T04: SUM 合并局部乘积；后续各 rank 操作同一个逻辑输出。

        clone contiguous 后在 TP group 求和。不能直接修改调用者的 partial。
        """
        todo("T04")

    @staticmethod
    def backward(ctx, grad):
        # TP replicas 执行同一份逻辑 loss，所以这里 identity，不再 all-reduce。
        return grad, None


def copy_to_tp(x, mesh):
    return _Copy.apply(x, mesh)


def reduce_from_tp(x, mesh):
    return _Reduce.apply(x, mesh)
