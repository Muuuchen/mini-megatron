"""共享的 param->flat->shard 元数据，ZeRO/FSDP/checkpoint 不各写一套切片。"""
from dataclasses import dataclass
import torch
from megatron_mini.exercise import todo

@dataclass(frozen=True)
class Slice:
    name: str
    shape: tuple
    start: int
    end: int

@dataclass(frozen=True)
class FlatLayout:
    entries: tuple
    numel: int
    padded_numel: int
    shard_numel: int
    world_size: int

    @classmethod
    def build(cls,named_tensors,world_size):
        """T35: named_tensors 是有序(name,tensor)列表，记录 shape/offset。
        padded_numel=ceil(numel/world_size)*world_size，支持参数跨 shard 边界。
        拒绝空参数、重复名字、共享 storage/tied 参数（不静默重复更新）。
        输出 FlatLayout；原始 shapes 不包含 padding。
        """
        todo('T35')

    def pack(self,named_tensors):
        tensors=dict(named_tensors)
        first=tensors[self.entries[0].name]
        flat=first.new_zeros(self.padded_numel)
        for e in self.entries: flat[e.start:e.end]=tensors[e.name].reshape(-1)
        return flat

    def views(self,flat):
        if flat.numel()!=self.padded_numel: raise ValueError('wrong flat length')
        return {e.name:flat[e.start:e.end].view(e.shape) for e in self.entries}

    def shard(self,flat,rank):
        if not 0<=rank<self.world_size: raise ValueError('bad rank')
        return flat[rank*self.shard_numel:(rank+1)*self.shard_numel].clone()
