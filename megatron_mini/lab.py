"""v2 独立数值实验和集成接线入口。 --list 不运行学生实现。"""
import argparse,json,os
from dataclasses import replace
from datetime import timedelta
import torch
import torch.distributed as dist
from megatron_mini.tasks import require_tasks,pending_tasks
from megatron_mini.distributed.axis import Axis
from megatron_mini.labs.catalog import CASES,RECIPES,recipe_tasks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list',action='store_true')
    p.add_argument('--case',choices=CASES)
    p.add_argument('--recipe',choices=RECIPES)
    p.add_argument('--run',action='store_true',help='actually run recipe, not just inspect plan')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--precision',choices=['fp32','bf16','fp16'],default=None)
    p.add_argument('--steps',type=int,default=3)
    a=p.parse_args()
    if a.case and a.recipe: p.error('choose either case or recipe')
    if a.steps<1: p.error('positive steps required')
    if a.list or (not a.case and not a.recipe):
        pending=pending_tasks()
        for name,ids in CASES.items():
            print(f'case {name:15} needs={",".join(ids) or "provided plumbing"} pending={len(set(ids)&pending.keys())}')
        for name,c in RECIPES.items():
            print(f'recipe {name:20} world={c.dims.world_size}; integration TODO; inspect with --recipe {name}')
        return
    c=replace(RECIPES[a.recipe],precision=a.precision or RECIPES[a.recipe].precision) if a.recipe else None
    if c:
        c.validate();ids=recipe_tasks(c)
        if not a.run:
            print(json.dumps({'recipe':a.recipe,'config':c.validate(),'required_tasks':ids,
                              'state':'planned integration, not implemented'},ensure_ascii=False,indent=2))
            return
    else: ids=CASES[a.case]
    require_tasks(ids)  # Must fail before init_process_group to avoid asymmetric rank hangs.
    world=int(os.environ.get('WORLD_SIZE','1'))
    if c and world!=c.dims.world_size: p.error(f'need WORLD_SIZE={c.dims.world_size}')
    device=torch.device(a.device)
    if a.device=='cuda':
        if not torch.cuda.is_available(): p.error('CUDA is unavailable')
        local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local);device=torch.device('cuda',local)
    if c and c.sharding=='torch_fsdp2' and device.type!='cuda': p.error('FSDP2 reference recipe needs CUDA')
    torch.set_num_threads(1);torch.manual_seed(123)
    created=False
    try:
        if world>1:
            dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(seconds=90));created=True
        axis=Axis.world()
        if a.case:
            from megatron_mini.labs.checks import run_case
            run_case(a.case,axis,device)
            if axis.rank==0: print(f'PASS {a.case}: numerical/contract checks, not a performance claim')
        else:
            from megatron_mini.training.engine_v2 import build_training_runtime,train_step_v2
            from megatron_mini.config import ModelConfig
            from megatron_mini.data import make_batch,split_microbatches
            cfg=ModelConfig();runtime=build_training_runtime(c,cfg,device)
            d,_,_,_=c.dims.coords(axis.rank)
            for step in range(a.steps):
                batch=make_batch(cfg,step,4,d,c.dp)
                metrics=train_step_v2(runtime,split_microbatches(batch,2))
                if axis.rank==0: print(metrics)
    finally:
        if created: dist.destroy_process_group()

if __name__=='__main__': main()
