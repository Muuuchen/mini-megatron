"""显式生命周期的教学 FSDP：forward释放全参，backward再聚合并重算。
这是 FSDP + 强制单元重算的简化算法，不是假称 PyTorch FSDP2 的 hook 实现。
不保存完整参数的计算图，避免做了 all-gather 却没有真正释放 full weights。
"""
import copy
import torch
from torch import nn
from torch.func import functional_call
from megatron_mini.exercise import todo
from megatron_mini.optimizer.flat import FlatLayout

class _ShardedUnit(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,local_shard,unit):
        """T38: gather local_shard 得到 flat full params -> functional_call。
        ctx 只保存 x/local_shard、unit和 CPU/CUDA RNG状态，不保存 full flat。
        记录 forward 前 RNG；backward重算时恢复但不扰动训练的当前 RNG。
        ctx 可保存 autocast配置供反向复现；baseline unit FP32。
        full 参数仅存在当前调用中；返回后必须不再被 unit/template/ctx 持有。
        """
        todo('T38')

    @staticmethod
    def backward(ctx,grad_output):
        """T39: 再 gather -> full.detach().requires_grad_()；浮点x detach+require；int64 token输入不求dx。
        enable_grad + fork_rng 重算 unit.call(x,full)，用 autograd.grad 得到
        dx 与 dflat。先沿 unit.grad_sum_axes SUM，再 RS(dflat)/unit.grad_divisor，
        返回(dx,dlocal,None)，整数输入dx=None。默认grad_divisor=axis.size。
        x 是各 DP rank 不同样本，dx 不做 DP 平均；dflat 才做平均。
        高阶导数不要求；不通过 p.data 替换 shape 欺骗 autograd。
        若外层已采用 sum/global_token_count，RS divisor 必须改为1（T49）。
        """
        todo('T39')

class FullShardUnit(nn.Module):
    def __init__(self,module,axis,grad_divisor=None,grad_sum_axes=()):
        super().__init__()
        if list(module.named_buffers()):
            raise ValueError('minimal FSDP unit disallows mutable/buffer state')
        self.axis=axis
        self.grad_divisor=float(axis.size if grad_divisor is None else grad_divisor)
        self.grad_sum_axes=tuple(grad_sum_axes)
        if self.grad_divisor<=0: raise ValueError("positive gradient divisor required")
        self.layout=FlatLayout.build(list(module.named_parameters()),axis.size)
        flat=self.layout.pack(list(module.named_parameters())).detach().float()
        self.local_shard=nn.Parameter(self.layout.shard(flat,axis.rank))
        # meta template仅描述计算结构，无 full-weight storage；不能交给optimizer。
        object.__setattr__(self,'template',copy.deepcopy(module).to('meta'))

    def call(self,x,flat):
        return functional_call(self.template,self.layout.views(flat),(x,),strict=True)

    def forward(self,x):
        return _ShardedUnit.apply(x,self.local_shard,self)

    @torch.no_grad()
    def full_state_dict(self):
        return {n:p.clone() for n,p in self.layout.views(self.axis.gather(self.local_shard)).items()}


def shard_blocks(module,axis):
    """T40: bottom-up 用 FullShardUnit 管理每个 transformer block或MLP block。
    剩余 embedding/norm/head 另成 units；只能在结构构造时做，optimizer 后建。
    不要将整个 GPT 聚为一个 full unit，否则无法观察层级 live memory 差异。
    单元必须是 tensor->tensor；GPT token lookup边界由 T55 adapter 明确处理。
    """
    todo('T40')
