import torch
from torch.nn import functional as F
from megatron_mini.exercise import todo


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """T01: logits=[B,S,V]，targets=[B,S]，返回所有 B*S token 的 mean CE。

    labels 已由 data.py 右移。不要再次 shift；这里没有 padding/ignore_index。
    logits 可能不连续，使用 reshape 而非假设 view 合法。
    返回保留计算图的标量，不能 .item()/detach()。
    """
    todo("T01")
