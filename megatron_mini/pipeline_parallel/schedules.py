"""GPipe fill-drain：每个 stage 先 F0..F(M-1)，再 B(M-1)..B0。

这是教学基线，不冒称 Megatron 的 1F1B；默认保留全部 microbatch 激活。
"""
from dataclasses import dataclass
import torch
from megatron_mini.exercise import todo
from megatron_mini.tensor_parallel.loss import language_model_loss
from .p2p import recv_forward, send_forward, recv_backward, send_backward


@dataclass
class Record:
    input_activation: torch.Tensor | None
    output: torch.Tensor
    loss: torch.Tensor | None


def gpipe_forward(stage, microbatches) -> list[Record]:
    """T16: forward 所有 microbatches，返回按 microbatch 升序保存的图记录。

    first: stage_input=input_ids，Record.input_activation=None；
    others: recv_forward(B,S,H)，保留叶子输入以便 backward 发回其 grad。
    执行 stage；非 last send_forward，last 计算未缩放 mean loss。
    output 必须保留本地计算图，只有通信 payload detach。
    不在 forward 阶段 backward，不做 optimizer.step。
    """
    if stage.mesh.dims.pp <= 1 or not microbatches:
        raise ValueError("GPipe requires PP>1 and nonempty microbatches")
    todo("T16")


def gpipe_backward(records: list[Record], mesh) -> torch.Tensor:
    """T17: 按 reversed(records) 回传梯度；返回最后 stage 的 detached 平均 loss。

    last: (record.loss/M).backward()；其他: grad=recv_backward(output)，
    torch.autograd.backward(output, grad)。非 first 把 input_activation.grad 发上游。
    microbatch 的 1/M 只在末 stage 加一次，其他 stage 不要再次缩放。
    非 last 返回 device 上的零标量，方便后续 PP SUM 聚合日志。
    结束时 records.clear() 释放持有的图引用；不要使用 retain_graph=True。
    """
    if not records:
        raise ValueError("records must not be empty")
    todo("T17")
