"""一个模型表示同时服务 TP、PP：PP=1 时就是完整的 TP GPT。"""
import torch
from torch import nn
from torch.nn import functional as F
from megatron_mini.exercise import todo
from megatron_mini.tensor_parallel.layers import (
    ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)
from .dense import causal_attention


class ParallelAttention(nn.Module):
    def __init__(self, cfg, mesh):
        super().__init__()
        self.local_heads = cfg.num_heads // mesh.dims.tp
        self.head_dim = cfg.head_dim
        self.q = ColumnParallelLinear(cfg.hidden_size, cfg.hidden_size, mesh)
        self.k = ColumnParallelLinear(cfg.hidden_size, cfg.hidden_size, mesh)
        self.v = ColumnParallelLinear(cfg.hidden_size, cfg.hidden_size, mesh)
        self.proj = RowParallelLinear(cfg.hidden_size, cfg.hidden_size, mesh)

    def forward(self, x):
        """T10: 按完整 attention head 切分，无 head 内部跨 rank 通信。

        Q/K/V 各自列并行 -> reshape [B,local_heads,S,D] -> causal_attention
        -> [B,S,H/TP] -> 行并行 proj -> replicated [B,S,H]。
        Q/K/V 独立参数，故没有 fused QKV 布局切错的问题。
        禁止把全局 num_heads 当作本地 head 数，禁止 all-gather Q/K/V。
        """
        todo("T10")


class ParallelMLP(nn.Module):
    def __init__(self, cfg, mesh):
        super().__init__()
        self.fc1 = ColumnParallelLinear(cfg.hidden_size, cfg.ffn_size, mesh)
        self.fc2 = RowParallelLinear(cfg.ffn_size, cfg.hidden_size, mesh)

    def forward(self, x):
        """T11: column -> GELU -> row，中间激活保持 [B,S,FF/TP]。

        GELU 使用 F.gelu 默认精确模式，与 dense oracle 一致。
        """
        todo("T11")


class ParallelBlock(nn.Module):
    def __init__(self, cfg, mesh):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = ParallelAttention(cfg, mesh)
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = ParallelMLP(cfg, mesh)

    def forward(self, x):
        """T12: 两段 pre-norm + residual，LayerNorm 参数在 TP 内复制。

        Block 边界 [B,S,H] 是 replicated，注意第二段 norm 的输入是更新后的 x。
        """
        todo("T12")


class GPTStage(nn.Module):
    def __init__(self, cfg, mesh):
        super().__init__()
        cfg.validate(mesh.dims.tp, mesh.dims.pp)
        self.cfg, self.mesh = cfg, mesh
        count = cfg.num_layers // mesh.dims.pp
        start = mesh.pp_rank * count
        # 每个 stage 只分配自己持有的层，禁止每个 stage 常驻完整模型。
        if mesh.first:
            self.tok = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size, mesh)
            self.pos = nn.Embedding(cfg.seq_len, cfg.hidden_size)
        self.blocks = nn.ModuleDict({str(i): ParallelBlock(cfg, mesh)
                                     for i in range(start, start + count)})
        if mesh.last:
            self.final_norm = nn.LayerNorm(cfg.hidden_size)
            self.head = ColumnParallelLinear(cfg.hidden_size, cfg.vocab_size, mesh, bias=False)

    def forward(self, stage_input):
        """T13: 首 stage 接 tokens:int64[B,S]；其他接 hidden:fp32[B,S,H]。

        first: tok + absolute pos embedding；所有 stage 执行各自 blocks；
        last: final_norm + head，输出 [B,S,V/TP]，否则输出 [B,S,H]。
        PP 中同一个 tp_rank 与下一 stage 相同 tp_rank 通信；边界激活完整复制。
        不在本函数中 send/recv，不计算 loss，也不 step optimizer。
        """
        todo("T13")
