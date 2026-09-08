import torch
from megatron_mini.exercise import todo
from megatron_mini.tensor_parallel.loss import language_model_loss


def forward_backward_no_pipeline(stage, microbatches) -> torch.Tensor:
    """T15: PP=1 的梯度累积；返回本 DP rank 的 detached 平均 loss。

    调用者已经 zero_grad；此函数不要 zero_grad / optimizer.step / DP all-reduce。
    每个 microbatch 先 forward -> mean token loss -> (loss / M).backward()。
    M=len(microbatches)，本 lab 的 microbatch 大小和序列长度固定相等。
    每个 microbatch 即时 backward，不要先攒完 M 个图。
    日志平均 loss 不额外除 DP；DP 日志规约由 engine 负责。
    """
    if stage.mesh.dims.pp != 1 or not microbatches:
        raise ValueError("no-pipeline schedule requires PP=1 and nonempty microbatches")
    todo("T15")
