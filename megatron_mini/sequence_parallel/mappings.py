"""这里不是旧 TP copy/reduce 的替代名字：sequence 布局改变了伴随算子。"""
import torch
from megatron_mini.exercise import todo

class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,axis):
        ctx.axis=axis
        return axis.gather(x,dim=1)
    @staticmethod
    def backward(ctx,grad):
        """T22: [B,S,H] partial dX -> SUM+sequence shard，不能只切片。"""
        todo('T22')

class _Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,axis):
        """T23: partial [B,S,H] -> SUM 后 [B,S/TP,H]，保存 axis。"""
        todo('T23')
    @staticmethod
    def backward(ctx,grad):
        return ctx.axis.gather(grad,dim=1),None

def gather_sequence(x,axis): return _Gather.apply(x,axis)
def scatter_sequence(x,axis): return _Scatter.apply(x,axis)
