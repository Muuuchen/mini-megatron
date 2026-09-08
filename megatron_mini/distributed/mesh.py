"""进程生命周期和真实 ProcessGroup 由脚手架提供，不用全局单例。"""
from dataclasses import dataclass, field
from datetime import timedelta
import os
import torch
import torch.distributed as dist
from .topology import ParallelDims, plan_groups


@dataclass
class Mesh:
    dims: ParallelDims
    rank: int
    device: torch.device
    groups: dict = field(default_factory=dict)
    members: dict = field(default_factory=dict)

    @property
    def dp_rank(self): return self.dims.coords(self.rank)[0]
    @property
    def pp_rank(self): return self.dims.coords(self.rank)[1]
    @property
    def tp_rank(self): return self.dims.coords(self.rank)[2]
    @property
    def first(self): return self.pp_rank == 0
    @property
    def last(self): return self.pp_rank == self.dims.pp - 1

    def group(self, axis): return self.groups.get(axis)
    def size(self, axis): return getattr(self.dims, axis)

    def peer(self, pp_rank):
        return self.dims.rank(self.dp_rank, pp_rank, self.tp_rank)

    @classmethod
    def local(cls, device="cpu"):
        return cls(ParallelDims(), 0, torch.device(device))


def initialize(dims: ParallelDims, device: str = "cpu") -> Mesh:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world != dims.world_size:
        raise ValueError(f"WORLD_SIZE={world}, but tp*pp*dp={dims.world_size}")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu for correctness labs")
        index = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(index)
        dev = torch.device("cuda", index)
        # 教学验收优先控制 FP32 数值；不承诺跨设备 bitwise。
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif device == "cpu":
        dev = torch.device("cpu")
    else:
        raise ValueError("device must be cpu or cuda")
    if world > 1:
        dist.init_process_group("nccl" if device == "cuda" else "gloo",
                                timeout=timedelta(seconds=90))
    mesh = Mesh(dims, rank, dev)
    if world > 1:
        # 所有 ranks 按同样顺序进入；singleton 不必真的 new_group。
        for spec in plan_groups(dims):
            group = dist.new_group(list(spec.ranks)) if len(spec.ranks) > 1 else None
            if rank in spec.ranks:
                mesh.groups[spec.axis] = group
                mesh.members[spec.axis] = spec.ranks
    return mesh


def shutdown():
    if dist.is_initialized():
        # 异常路径不加 barrier，避免失败 rank 等不到其他 rank。
        dist.destroy_process_group()


def sum_(tensor, mesh: Mesh, axis: str):
    """原地 SUM：只用于显式非自动微分的通信或自定义 autograd 内部。"""
    if mesh.size(axis) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=mesh.group(axis))
    return tensor


def reported_loss(local_loss: torch.Tensor, mesh: Mesh):
    # 只有最后一个 PP stage 提供 loss；TP 副本已有相同 loss，不再 TP 求和。
    value = local_loss.detach().clone()
    sum_(value, mesh, "pp")
    sum_(value, mesh, "dp")
    return value / mesh.dims.dp
