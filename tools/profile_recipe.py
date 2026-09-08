"""GPU-only profile harness; requires T55 and all selected recipe tasks. No claimed speedup."""
import argparse,os,time
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from megatron_mini.labs.catalog import RECIPES,recipe_tasks
from megatron_mini.tasks import require_tasks
from megatron_mini.config import ModelConfig
from megatron_mini.data import make_batch,split_microbatches
from megatron_mini.training.engine_v2 import build_training_runtime,train_step_v2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',choices=RECIPES,required=True)
    p.add_argument('--trace-dir',default='runs/traces')
    p.add_argument('--steps',type=int,default=5)
    a=p.parse_args();c=RECIPES[a.recipe];c.validate();require_tasks(recipe_tasks(c))
    if a.steps<1: p.error('positive steps required')
    if not torch.cuda.is_available(): p.error('CUDA required to claim physical overlap')
    if int(os.environ.get('WORLD_SIZE','1'))!=c.dims.world_size: p.error('wrong world size')
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    try:
        cfg=ModelConfig();rank=dist.get_rank();d,_,_,_=c.dims.coords(rank)
        runtime=build_training_runtime(c,cfg,device)
        def step(i):
            batch=make_batch(cfg,i,4,d,c.dp)
            with torch.profiler.record_function('logical_train_step'):
                return train_step_v2(runtime,split_microbatches(batch,2))
        for i in range(3): step(i)
        torch.cuda.synchronize();dist.barrier();torch.cuda.reset_peak_memory_stats()
        elapsed=[]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True,profile_memory=True) as prof:
            for i in range(a.steps):
                torch.cuda.synchronize();start=time.perf_counter();step(i+3);torch.cuda.synchronize()
                elapsed.append(time.perf_counter()-start);prof.step()
        out=Path(a.trace_dir);out.mkdir(parents=True,exist_ok=True)
        prof.export_chrome_trace(str(out/f'{a.recipe}.rank{rank}.json'))
        stats=torch.tensor([sum(elapsed)/len(elapsed),torch.cuda.max_memory_allocated()],device=device,dtype=torch.float64)
        dist.all_reduce(stats,op=dist.ReduceOp.MAX)
        if rank==0:
            print({'max_rank_step_seconds':stats[0].item(),'max_rank_peak_allocated_bytes':int(stats[1]),
                   'global_tokens_per_step':4*cfg.seq_len*c.dp,
                   'note':'profiler perturbs timing; compare unprofiled serial/overlap runs separately'})
    finally: dist.destroy_process_group()

if __name__=='__main__': main()
