"""PyTorch weight=[out_features,in_features]，注意与论文 W 的记号方向。"""
import torch
from torch import nn
from torch.nn import functional as F
from megatron_mini.exercise import todo
from .mappings import copy_to_tp, reduce_from_tp


def parameter(shape, shard_dim=None):
    p = nn.Parameter(torch.empty(shape))
    nn.init.normal_(p, std=0.02)
    # 元数据用于初始化切片、梯度核验、TP 去重的 global grad norm。
    p._shard_dim = shard_dim
    return p


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, mesh, bias=True):
        super().__init__()
        if out_features % mesh.dims.tp:
            raise ValueError("out_features must be divisible by TP")
        self.mesh = mesh
        self.weight = parameter((out_features // mesh.dims.tp, in_features), 0)
        self.bias = parameter((out_features // mesh.dims.tp,), 0) if bias else None

    def forward(self, x):
        """T05: [B,S,Hin] replicated -> [B,S,Hout/TP] sharded。

        输入经过 copy_to_tp，再使用本地 F.linear；不 gather 输出。
        bias 按输出维切片，各 rank 独有。为何普通 F.linear 不足以得到正确 dX？
        """
        todo("T05")


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, mesh, bias=True):
        super().__init__()
        if in_features % mesh.dims.tp:
            raise ValueError("in_features must be divisible by TP")
        self.mesh = mesh
        self.weight = parameter((out_features, in_features // mesh.dims.tp), 1)
        self.bias = parameter((out_features,)) if bias else None

    def forward(self, local_x):
        """T06: [B,S,Hin/TP] sharded -> [B,S,Hout] replicated。

        本地无 bias matmul -> reduce_from_tp -> 最后加 replicated bias。
        不能先加 bias 再 SUM，否则 bias 变成 TP 倍。输入已经分片，不要再 split。
        """
        todo("T06")


class VocabParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, mesh):
        super().__init__()
        if vocab_size % mesh.dims.tp:
            raise ValueError("vocab_size must be divisible by TP")
        self.mesh = mesh
        local_vocab = vocab_size // mesh.dims.tp
        self.start = mesh.tp_rank * local_vocab
        self.end = self.start + local_vocab
        self.weight = parameter((local_vocab, hidden_size), 0)

    def forward(self, tokens):
        """T07: 词表分片；tokens 全局 ID -> replicated [B,S,H]。

        先构造范围 mask，域外 token 安全映射到本地 index=0，再 embedding。
        把域外向量乘成 0，最后 reduce_from_tp。禁止负索引和越界索引。
        mask 必须屏蔽域外 token 对第 0 行 embedding 的梯度。
        """
        todo("T07")
