from megatron_mini.exercise import todo
from .mappings import gather_sequence,scatter_sequence

def sp_mlp(x,norm,w1,b1,w2,b2,axis):
    """T24: x[B,S/TP,H], w1[F/TP,H], w2[H,F/TP] -> 与 x 同形。
    顺序: local norm -> AG -> local F.linear(w1,b1) -> GELU ->
    local F.linear(w2,None) -> RS -> +b2 -> +local residual。
    不可直接调用旧 ColumnParallelLinear: 它会重复 all-reduce dX。
    b2 在 RS 后添加，只添加一次。norm/b2 的梯度在 T25 做 TP SUM。
    验收: lab sp，前向、local dX、两种分片权重及复制 LN/bias 梯度。
    """
    todo('T24')

def sync_sp_replicated(parameters,axis):
    """T25: 只同步 sequence-local norm/bias 参数的 grad，SUM 不平均。
    先有完整名称/角色清单再遍历，不要对 TP-sharded GEMM 参数做 SUM。
    同步在所有 microbatch 完成后，和 DP reduction 是不同的维度。
    """
    todo('T25')
