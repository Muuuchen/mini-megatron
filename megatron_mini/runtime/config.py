"""课程配置与拒绝静默降级。配置通过只代表可规划，不代表 TODO 已完成。"""
from dataclasses import dataclass, asdict, field
from megatron_mini.distributed.topology_v2 import MeshDims
from megatron_mini.offload.config import OffloadConfig

@dataclass(frozen=True)
class RecipeConfig:
    tp: int=1
    pp: int=1
    dp: int=1
    cp: int=1
    ep: int=1
    sp: bool=False
    sharding: str='ddp'
    schedule: str='gpipe'
    virtual_stages: int=1
    precision: str='fp32'
    recompute: str='none'
    cp_mode: str='gather'
    experts: int=4
    topk: int=2
    model: str='gpt'
    overlap: tuple=()
    offload: OffloadConfig=field(default_factory=OffloadConfig)

    @property
    def dims(self): return MeshDims(self.tp,self.pp,self.dp,self.cp,self.ep)

    def validate(self):
        self.dims
        if self.sharding not in ('ddp','zero1','zero2','zero3','fsdp','torch_fsdp2'):
            raise ValueError('sharding must be ddp/zero1/zero2/zero3/fsdp/torch_fsdp2')
        if self.precision not in ('fp32','bf16','fp16'): raise ValueError('FP8 is excluded')
        if self.schedule not in ('gpipe','1f1b','interleaved'): raise ValueError('unknown schedule')
        if self.recompute not in ('none','block','attention','sac'): raise ValueError('unknown recompute')
        if self.model not in ('gpt','llama','moe'): raise ValueError('unknown model')
        if self.cp_mode not in ('gather','ring'): raise ValueError('unknown CP mode')
        if self.sp and self.tp<2: raise ValueError('SP requires TP > 1 in this lab')
        if self.ep>1 and self.model!='moe': raise ValueError('EP requires model=moe')
        if self.ep>1 and self.tp>1 and not self.sp: raise ValueError('TP+EP requires SP here')
        if self.experts % self.ep or not 1<=self.topk<=self.experts: raise ValueError('invalid experts/topk')
        if self.virtual_stages < 1: raise ValueError('virtual_stages must be positive')
        if self.virtual_stages>1 and self.schedule!='interleaved': raise ValueError('VPP requires interleaved')
        if self.schedule=='interleaved' and (self.pp<2 or self.virtual_stages<2):
            raise ValueError('interleaved requires PP>=2 and virtual_stages>=2')
        if set(self.overlap)-{'dp','tp','pp','ep','cp','fsdp'}: raise ValueError('unknown overlap axis')
        self.offload.validate(self.sharding)
        return asdict(self)
