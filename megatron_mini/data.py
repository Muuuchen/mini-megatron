"""可复现的合成 next-token 任务：无下载、tokenizer、外部数据依赖。"""
from dataclasses import dataclass
import torch
from .config import ModelConfig


@dataclass
class Batch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor


def make_batch(cfg: ModelConfig, step: int, local_batch: int,
               dp_rank: int = 0, dp_size: int = 1, device="cpu") -> Batch:
    if step < 0 or local_batch < 1 or not 0 <= dp_rank < dp_size:
        raise ValueError("invalid step/batch/DP coordinate")
    # 样本 ID 在 DP ranks 之间不重叠，TP/PP ranks 使用完全相同的 DP 样本。
    ids = step * local_batch * dp_size + dp_rank * local_batch + torch.arange(local_batch)
    starts = (ids * 7 + cfg.seed) % cfg.vocab_size
    tokens = (starts[:, None] + torch.arange(cfg.seq_len + 1)[None, :]) % cfg.vocab_size
    tokens = tokens.to(device=device, dtype=torch.long)
    # 已右移好 labels，loss 内不可以再次 shift。
    return Batch(tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous())


def split_microbatches(batch: Batch, micro_batch: int) -> list[Batch]:
    b = batch.input_ids.shape[0]
    if micro_batch < 1 or b % micro_batch:
        raise ValueError("local batch must be divisible by micro_batch")
    return [Batch(x, y) for x, y in zip(batch.input_ids.split(micro_batch),
                                       batch.target_ids.split(micro_batch))]
