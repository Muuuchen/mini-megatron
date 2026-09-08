"""配置通过只表示课程允许规划，不表示组合已经接线或完成 GPU 验收。"""
from dataclasses import dataclass

@dataclass(frozen=True)
class OffloadConfig:
    optimizer: str = 'none'  # includes local FP32 master / moments / reduced gradients
    parameters: str = 'none' # persistent local parameter shard, not full model
    activations: str = 'none'
    pin_memory: bool = False
    async_transfer: bool = False

    @property
    def enabled(self):
        return any(x == 'cpu' for x in (self.optimizer, self.parameters, self.activations))

    def validate(self, sharding):
        if any(x not in ('none','cpu') for x in (self.optimizer,self.parameters,self.activations)):
            raise ValueError('this course implements none/cpu offload; NVMe is a documented extension')
        if self.parameters == 'cpu':
            if sharding not in ('zero3','fsdp','torch_fsdp2'):
                raise ValueError('parameter offload requires a fully sharded strategy in this course')
            if self.optimizer != 'cpu':
                raise ValueError('mini CPU parameter ownership requires CPU optimizer too')
        if self.optimizer == 'cpu' and sharding == 'ddp':
            raise ValueError('CPU optimizer lab targets ZeRO/FSDP shards, not full DDP replicas')
        if sharding == 'torch_fsdp2' and self.optimizer != self.parameters:
            raise ValueError('FSDP2 CPUOffloadPolicy is not an optimizer-only offload switch')
        if self.async_transfer and (not self.pin_memory or not self.enabled):
            raise ValueError('asynchronous CPU transfers need pinned buffers and enabled offload')
        return self
