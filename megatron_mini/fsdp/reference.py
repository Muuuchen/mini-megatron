"""已给出的官方 FSDP2 对照入口；不能用它绕过 T38/T39 手写练习。"""
import torch

def wrap_torch_fsdp2(model,mesh,blocks,precision='fp32',cpu_offload=False):
    if not torch.cuda.is_available():
        raise RuntimeError('this reference example is CUDA-only; use mini FSDP CPU lab for semantics')
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    dtype={'fp32':None,'bf16':torch.bfloat16,'fp16':torch.float16}[precision]
    policy=MixedPrecisionPolicy(param_dtype=dtype,reduce_dtype=torch.float32)
    kwargs={}
    if cpu_offload:
        from torch.distributed.fsdp import CPUOffloadPolicy
        kwargs['offload_policy']=CPUOffloadPolicy(pin_memory=True)
    for block in blocks:
        fully_shard(block,mesh=mesh,reshard_after_forward=True,mp_policy=policy,**kwargs)
    fully_shard(model,mesh=mesh,reshard_after_forward=False,mp_policy=policy,**kwargs)
    return model  # optimizer MUST be constructed after wrapping
