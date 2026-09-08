"""已提供的原语适配。这里只做传输，不隐含 autograd 或梯度平均。"""
from dataclasses import dataclass
import torch
import torch.distributed as dist

@dataclass(frozen=True)
class Axis:
    group: object = None
    size: int = 1
    rank: int = 0

    @classmethod
    def world(cls):
        if not dist.is_initialized(): return cls()
        return cls(dist.group.WORLD, dist.get_world_size(), dist.get_rank())

    def sum(self, x):
        out = x.contiguous().clone()
        if self.size > 1: dist.all_reduce(out, group=self.group)
        return out

    def gather(self, x, dim=0):
        """只接受等尺寸分片；返回按 group rank 排序的拼接。"""
        if self.size == 1: return x.clone()
        parts = [torch.empty_like(x) for _ in range(self.size)]
        dist.all_gather(parts, x.contiguous(), group=self.group)
        return torch.cat(parts, dim=dim)

    def reduce_scatter(self, x, dim=0, divisor=1.0):
        if x.shape[dim] % self.size: raise ValueError('uneven reduce-scatter')
        if divisor <= 0: raise ValueError('divisor must be positive')
        if self.size == 1: return x.clone() / divisor
        if dist.get_backend(self.group) == 'nccl':
            src = x.movedim(dim, 0).contiguous()
            out = torch.empty((src.shape[0]//self.size, *src.shape[1:]), device=x.device, dtype=x.dtype)
            dist.reduce_scatter_tensor(out, src, group=self.group)
            return out.movedim(0, dim).contiguous() / divisor
        # Gloo portability path: numerical SUM+slice, NOT physical RS memory/performance.
        return self.sum(x).chunk(self.size, dim=dim)[self.rank].contiguous() / divisor
