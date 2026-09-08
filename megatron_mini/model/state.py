"""用同一个 dense seed 初始化每个 shard，避免独立随机分片导致伪数值差异。"""
import torch
from .dense import build_dense
from .parallel import GPTStage


def shard_like(full_tensor, local_parameter, mesh):
    axis = getattr(local_parameter, "_shard_dim", None)
    if axis is None:
        return full_tensor
    return full_tensor.chunk(mesh.dims.tp, dim=axis)[mesh.tp_rank].contiguous()


@torch.no_grad()
def load_dense_state(stage, dense_state):
    for name, parameter in stage.named_parameters():
        full = dense_state[name]
        part = shard_like(full, parameter, stage.mesh)
        if part.shape != parameter.shape:
            raise ValueError(f"wrong shard shape for {name}: {part.shape} vs {parameter.shape}")
        parameter.copy_(part.to(device=parameter.device, dtype=parameter.dtype))


def build_stage(cfg, mesh):
    stage = GPTStage(cfg, mesh).to(mesh.device)
    # 教学捷径：每个 rank 在 CPU 临时构建 tiny dense 模型，仅用于初始化。
    # 大模型生产实现不能这样初始化；这里退出函数即释放 dense，不在 GPU 常驻副本。
    reference = build_dense(cfg)
    load_dense_state(stage, reference.state_dict())
    return stage
