"""CUDA-only FSDP2 reference with dense update comparison. Does not fill student TODOs.
Run from repository root: torchrun --standalone --nproc_per_node=2 -m examples.torch_fsdp2_reference
"""
import argparse,os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.nn import functional as F
from megatron_mini.config import ModelConfig
from megatron_mini.model.dense import build_dense
from megatron_mini.data import make_batch
from megatron_mini.fsdp.reference import wrap_torch_fsdp2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--cpu-offload',action='store_true')
    p.add_argument('--steps',type=int,default=3)
    a=p.parse_args()
    if not torch.cuda.is_available(): p.error('CUDA required; this example was not GPU-tested in delivery environment')
    if 'RANK' not in os.environ: p.error('launch this example with torchrun')
    if a.steps<1: p.error('steps must be positive')
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    try:
        world,rank=dist.get_world_size(),dist.get_rank()
        cfg=ModelConfig(num_layers=2)
        model=build_dense(cfg,torch.device('cpu') if a.cpu_offload else device);reference=build_dense(cfg,device)
        mesh=init_device_mesh('cuda',(world,),mesh_dim_names=('dp',))
        model=wrap_torch_fsdp2(model,mesh,list(model.blocks.values()),a.precision,cpu_offload=a.cpu_offload)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        ref_opt=torch.optim.AdamW(reference.parameters(),lr=1e-3)
        # Reference holds a dense model intentionally for numerical comparison, not memory benchmarking.
        for step in range(a.steps):
            optimizer.zero_grad(set_to_none=True);ref_opt.zero_grad(set_to_none=True)
            batch=make_batch(cfg,step,2,rank,world,device)
            global_batch=make_batch(cfg,step,2*world,0,1,device)
            logits=model(batch.input_ids)
            loss=F.cross_entropy(logits.float().reshape(-1,cfg.vocab_size),batch.target_ids.reshape(-1))
            loss.backward();optimizer.step()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.precision=='bf16'):
                reference_logits=reference(global_batch.input_ids)
            ref_loss=F.cross_entropy(reference_logits.float().reshape(-1,cfg.vocab_size),global_batch.target_ids.reshape(-1))
            ref_loss.backward();ref_opt.step()
            tolerance=(2e-2,2e-3) if a.precision=='bf16' else (1e-4,1e-5)
            ref_params=dict(reference.named_parameters())
            for name,param in model.named_parameters():
                # Inspection all-gather must use CUDA staging when the persistent shard is on CPU.
                inspect_param=param.detach().to(device)
                full=inspect_param.full_tensor() if hasattr(inspect_param,'full_tensor') else inspect_param
                torch.testing.assert_close(full.to(device),ref_params[name],rtol=tolerance[0],atol=tolerance[1])
            logged=loss.detach().clone();dist.all_reduce(logged);logged/=world
            if rank==0: print(f'step={step} mean_loss={logged.item():.6f}; update comparison passed')
    finally: dist.destroy_process_group()

if __name__=='__main__': main()
