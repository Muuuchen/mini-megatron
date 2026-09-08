import torch
from megatron_mini.exercise import todo

class _GatherKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,axis):
        ctx.axis=axis
        return axis.gather(x,dim=2)
    @staticmethod
    def backward(ctx,grad):
        """T26: 不同 query shard 对全局 KV 的梯度贡献 SUM+scatter，dim=2。"""
        todo('T26')

def gather_attention(q,k,v,axis):
    """T27: q/k/v[B,heads,S/CP,D] -> local output，按 contiguous 序列切分。
    用 _GatherKV 只聚合 K/V，Q 不聚合。mask 用全局 query offset:
    qpos=rank*S_local+arange(S_local)，允许 kpos<=qpos。
    logits FP32（float64 测试保持 float64），softmax 后乘 V。
    这里物化完整 KV，是 CP 的 correctness baseline，不宣称 ring 内存收益。
    验收: lab cp，跨 rank causal mask、dQ/dK/dV。
    """
    todo('T27')
