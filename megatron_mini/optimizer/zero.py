import torch
from torch import nn
from megatron_mini.exercise import todo
from .flat import FlatLayout

class ShardedOptimizer:
    """教学 ZeRO-1/2：模型参数保持复制，optimizer只拥有 local flat shard。
    注意: 直接从 param.grad flatten 的 ZeRO-2 教学路径会短暂保留完整梯度。
    若要持续降低 gradient memory，T34 必须接 gradient bucket hooks，及时释放
    完成 RS 的 full bucket；仅在 step 末尾 slice 不能宣称完整 ZeRO-2 内存收益。
    """
    def __init__(self,module,axis,stage=1,lr=1e-3):
        if stage not in (1,2): raise ValueError('stage3 is the FSDP path, not this optimizer')
        self.module,self.axis,self.stage=module,axis,stage
        self.layout=FlatLayout.build(list(module.named_parameters()),axis.size)
        flat=self.layout.pack(list(module.named_parameters())).detach().float()
        self.local=nn.Parameter(self.layout.shard(flat,axis.rank))
        self.optim=torch.optim.AdamW([self.local],lr=lr)

    def zero_grad(self):
        self.module.zero_grad(set_to_none=True);self.optim.zero_grad(set_to_none=True)

    def full_grad(self):
        params=list(self.module.named_parameters())
        if any(p.grad is None for _,p in params): raise RuntimeError('missing gradient')
        return self.layout.pack([(n,p.grad.float()) for n,p in params])

    @torch.no_grad()
    def copy_back(self,flat):
        views=self.layout.views(flat)
        for n,p in self.module.named_parameters(): p.copy_(views[n])

    def step(self):
        return self.step_zero1() if self.stage==1 else self.step_zero2()

    def step_zero1(self):
        """T36: full grad -> DP allreduce/DP -> local shard.grad -> local AdamW
        -> allgather updated local params -> copy_back。每步只更新一次。
        先不做 clip，由统一 engine 在正确时间处理。验收3次更新和 moments。
        """
        todo('T36')

    def step_zero2(self):
        """T37: full/bucket grad -> reduce_scatter /DP -> local AdamW
        -> allgather params。禁止对 RS 后 shard 再做一次 DP average。
        用全长瞬时 buffer 做数值 baseline，但内存实验必须接 T34 分桶。
        """
        todo('T37')
