"""完整、可运行的数值 oracle；不包含任何模型并行逻辑。"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from megatron_mini.config import ModelConfig


def causal_attention(q, k, v):
    """[B, heads, S, D] -> 同形状；显式 eager 算子方便阅读和对照。"""
    length = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    future = torch.ones(length, length, device=q.device, dtype=torch.bool).triu(1)
    return scores.masked_fill(future, float("-inf")).softmax(-1) @ v


class DenseAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.num_heads
        self.q = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.k = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.v = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x):
        b, s, h = x.shape
        q, k, v = [layer(x).reshape(b, s, self.heads, h // self.heads).transpose(1, 2)
                   for layer in (self.q, self.k, self.v)]
        y = causal_attention(q, k, v).transpose(1, 2).contiguous().reshape(b, s, h)
        return self.proj(y)


class DenseMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.ffn_size)
        self.fc2 = nn.Linear(cfg.ffn_size, cfg.hidden_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DenseBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = DenseAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = DenseMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class DenseGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos = nn.Embedding(cfg.seq_len, cfg.hidden_size)
        self.blocks = nn.ModuleDict({str(i): DenseBlock(cfg) for i in range(cfg.num_layers)})
        self.final_norm = nn.LayerNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # 不做 embedding/head weight tying，避免跨 PP stage 的共享参数归约。

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.tok(tokens) + self.pos(positions)
        for block in self.blocks.values():
            x = block(x)
        return self.head(self.final_norm(x))


def build_dense(cfg: ModelConfig, device="cpu") -> DenseGPT:
    # 初始化只消耗局部 CPU RNG 状态；所有 rank 同 seed 得到相同完整参数。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        model = DenseGPT(cfg)
    return model.to(device)
