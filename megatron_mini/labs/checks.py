"""独立数值oracle。这里不实现student算法；验收值、梯度、连续参数更新。
每个case在1/2个CPU rank即可起步。Transport case是已完成的底层脚手架。
"""
import copy
import tempfile
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from megatron_mini.distributed.axis import Axis


def close(a,b): torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)

def _input(rank,device):
    x=torch.arange(16,device=device,dtype=torch.float32).reshape(2,8)/16 + rank*0.15
    return x,x.sin()[:,:4]

def _mlp(device):
    return nn.Sequential(nn.Linear(8,12),nn.GELU(),nn.Linear(12,4)).to(device)


def transport(axis,device):
    x=torch.full((2,4),float(axis.rank+1),device=device)
    close(axis.sum(x),torch.full_like(x,axis.size*(axis.size+1)/2))
    gathered=axis.gather(x,1)
    assert gathered.shape==(2,4*axis.size)
    close(axis.reduce_scatter(gathered,1),x*axis.size)
    if axis.size>1:
        from megatron_mini.pipeline_parallel.transport import TypedP2P
        channel=TypedP2P(x.shape,x.dtype,device)
        h=channel.send(x,(axis.rank+1)%axis.size)
        recv=channel.recv((axis.rank-1)%axis.size)
        channel.finish()
        close(recv,torch.full_like(x,float((axis.rank-1)%axis.size+1)))
        assert h.finished


def topology(axis,device):
    from megatron_mini.distributed.topology_v2 import MeshDims,plan_groups_v2
    dims=MeshDims(tp=2,pp=2,dp=4,cp=2,ep=2)
    groups=plan_groups_v2(dims)
    order=['tp','pp','cp','dp','ep','edp','dp_cp','edp_cp']
    assert list(dict.fromkeys(g.axis for g in groups))==order
    expected={'tp':(0,1),'pp':(0,4),'cp':(0,2),'dp':(0,8,16,24),
              'ep':(0,8),'edp':(0,16),'dp_cp':(0,2,8,10,16,18,24,26),
              'edp_cp':(0,2,16,18)}
    for ax in order:
        lists=[g.ranks for g in groups if g.axis==ax]
        assert lists==sorted(lists)
        assert [x for x in lists if 0 in x]==[expected[ax]]
        for r in range(dims.world_size): assert sum(r in x for x in lists)==1


def sp(axis,device):
    from megatron_mini.sequence_parallel.block import sp_mlp,sync_sp_replicated
    if 8%axis.size or 12%axis.size: raise ValueError('SP lab requires size 1/2/4')
    full_x=torch.randn(2,8,8,device=device,requires_grad=True)
    norm=nn.LayerNorm(8).to(device);local_norm=copy.deepcopy(norm)
    up=nn.Linear(8,12).to(device);down=nn.Linear(12,8).to(device)
    x=full_x.detach().chunk(axis.size,dim=1)[axis.rank].clone().requires_grad_()
    w1=up.weight.detach().chunk(axis.size,0)[axis.rank].clone().requires_grad_()
    b1=up.bias.detach().chunk(axis.size,0)[axis.rank].clone().requires_grad_()
    w2=down.weight.detach().chunk(axis.size,1)[axis.rank].clone().requires_grad_()
    b2=down.bias.detach().clone().requires_grad_()
    ref=full_x+down(F.gelu(up(norm(full_x))))
    out=sp_mlp(x,local_norm,w1,b1,w2,b2,axis)
    close(out,ref.chunk(axis.size,1)[axis.rank])
    upstream=torch.randn_like(ref)
    (ref*upstream).sum().backward()
    (out*upstream.chunk(axis.size,1)[axis.rank]).sum().backward()
    sync_sp_replicated([*local_norm.parameters(),b2],axis)
    close(x.grad,full_x.grad.chunk(axis.size,1)[axis.rank])
    close(w1.grad,up.weight.grad.chunk(axis.size,0)[axis.rank])
    close(b1.grad,up.bias.grad.chunk(axis.size,0)[axis.rank])
    close(w2.grad,down.weight.grad.chunk(axis.size,1)[axis.rank])
    close(b2.grad,down.bias.grad)
    for p,q in zip(local_norm.parameters(),norm.parameters()): close(p.grad,q.grad)


def _cp(axis,device,use_ring=False):
    from megatron_mini.model.dense import causal_attention
    from megatron_mini.context_parallel.attention import gather_attention
    from megatron_mini.context_parallel.ring import ring_attention
    if 8%axis.size: raise ValueError('CP lab size must divide 8')
    full=[torch.randn(1,2,8,4,device=device,requires_grad=True) for _ in range(3)]
    local=[x.detach().chunk(axis.size,2)[axis.rank].clone().requires_grad_() for x in full]
    ref=causal_attention(*full)
    out=(ring_attention if use_ring else gather_attention)(*local,axis)
    close(out,ref.chunk(axis.size,2)[axis.rank])
    up=torch.randn_like(ref)
    (ref*up).sum().backward();(out*up.chunk(axis.size,2)[axis.rank]).sum().backward()
    for x,y in zip(local,full): close(x.grad,y.grad.chunk(axis.size,2)[axis.rank])

def cp(axis,device): _cp(axis,device)
def ring(axis,device): _cp(axis,device,True)


def _pipeline(axis,device,chunks=1):
    from megatron_mini.pipeline_parallel.planner import plan_1f1b,plan_interleaved,validate_schedule_simulation
    from megatron_mini.pipeline_parallel.executor import execute_events,PipelineIO
    from megatron_mini.pipeline_parallel.transport import TypedP2P
    if chunks>1 and axis.size<2: raise ValueError('interleaved lab requires >=2 processes')
    stages=axis.size;M=4
    if chunks>1 and M%stages: raise ValueError('interleaved lab requires process count dividing 4')
    planner=lambda p: (plan_1f1b(p,stages,M) if chunks==1 else plan_interleaved(p,stages,M,chunks))
    plans=[planner(p) for p in range(stages)]
    validate_schedule_simulation(plans,stages,M,chunks)
    full=nn.Sequential(*[nn.Sequential(nn.Linear(4,4),nn.GELU()) for _ in range(stages*chunks)]).to(device)
    local=[copy.deepcopy(full[c*stages+axis.rank]) for c in range(chunks)]
    batches=[torch.randn(2,4,device=device) for _ in range(M)]
    channel=TypedP2P((2,4),torch.float32,device)
    def first(c): return c*stages+axis.rank==0
    def last(c): return c*stages+axis.rank==stages*chunks-1
    def receive(direction,m,c):
        if direction=='forward':
            if first(c): return batches[m]
            return channel.recv((axis.rank-1)%stages,True)
        return channel.recv((axis.rank+1)%stages)
    def forward(m,c,x):
        y=local[c](x)
        return y,(y.square().mean() if last(c) else None)
    def send(direction,m,c,x):
        peer=(axis.rank+1)%stages if direction=='forward' else (axis.rank-1)%stages
        return channel.send(x,peer)
    io=PipelineIO(forward,receive,send,channel.finish,first,last)
    execute_events(plans[axis.rank],io,M)
    ref_loss=sum(full(x).square().mean() for x in batches)/M
    ref_loss.backward()
    for c,block in enumerate(local):
        for p,q in zip(block.parameters(),full[c*stages+axis.rank].parameters()): close(p.grad,q.grad)


def pipeline(axis,device): _pipeline(axis,device)
def interleaved(axis,device): _pipeline(axis,device,2)


def recompute(axis,device):
    from megatron_mini.runtime.recompute import recomputed
    model=nn.Sequential(nn.Linear(8,8),nn.Dropout(0.3),nn.GELU()).to(device)
    ref=copy.deepcopy(model);x=torch.randn(2,8,device=device)  # input intentionally has no grad
    torch.manual_seed(999);out=recomputed(model,x,enabled=True);out.square().sum().backward()
    after=torch.get_rng_state().clone()
    torch.manual_seed(999);expected=ref(x);expected.square().sum().backward()
    close(out,expected);assert torch.equal(after,torch.get_rng_state())
    for p,q in zip(model.parameters(),ref.parameters()): close(p.grad,q.grad)


def flat(axis,device):
    from megatron_mini.optimizer.flat import FlatLayout
    tensors=[('a',torch.arange(7.,device=device)),('b',torch.arange(6.,device=device).view(2,3))]
    layout=FlatLayout.build(tensors,4)
    assert (layout.numel,layout.padded_numel,layout.shard_numel)==(13,16,4)
    flat=layout.pack(tensors);assert torch.equal(flat[13:],torch.zeros_like(flat[13:]))
    for n,x in tensors: close(layout.views(flat)[n],x)
    close(torch.cat([layout.shard(flat,r) for r in range(4)]),flat)


def _zero(axis,device,stage):
    from megatron_mini.optimizer.zero import ShardedOptimizer
    model=_mlp(device);ref=copy.deepcopy(model)
    opt=ShardedOptimizer(model,axis,stage,lr=0.01);oracle=torch.optim.AdamW(ref.parameters(),lr=0.01)
    x,y=_input(axis.rank,device)
    all_data=[_input(r,device) for r in range(axis.size)]
    gx=torch.cat([v[0] for v in all_data]);gy=torch.cat([v[1] for v in all_data])
    for _ in range(3):
        opt.zero_grad();oracle.zero_grad(set_to_none=True)
        F.mse_loss(model(x),y).backward();F.mse_loss(ref(gx),gy).backward()
        opt.step();oracle.step()
        for p,q in zip(model.parameters(),ref.parameters()): close(p,q)
        for key in ('exp_avg','exp_avg_sq'):
            ref_flat=opt.layout.pack([(n,oracle.state[p][key]) for n,p in ref.named_parameters()])
            close(opt.optim.state[opt.local][key],opt.layout.shard(ref_flat,axis.rank))

def zero1(axis,device): _zero(axis,device,1)
def zero2(axis,device): _zero(axis,device,2)


def fsdp(axis,device):
    from megatron_mini.fsdp.unit import FullShardUnit
    model=_mlp(device);ref=copy.deepcopy(model)
    unit=FullShardUnit(model,axis)
    assert list(dict(unit.named_parameters()))==['local_shard']
    assert all(p.is_meta for p in unit.template.parameters())
    opt=torch.optim.AdamW(unit.parameters(),lr=0.01);oracle=torch.optim.AdamW(ref.parameters(),lr=0.01)
    x,y=_input(axis.rank,device)
    all_data=[_input(r,device) for r in range(axis.size)]
    gx=torch.cat([v[0] for v in all_data]);gy=torch.cat([v[1] for v in all_data])
    for _ in range(3):
        opt.zero_grad(set_to_none=True);oracle.zero_grad(set_to_none=True)
        local_x=x.clone().requires_grad_()
        out=unit(local_x);close(out,ref(x))
        F.mse_loss(out,y).backward();F.mse_loss(ref(gx),gy).backward()
        expected=unit.layout.pack([(n,p.grad) for n,p in ref.named_parameters()])
        close(unit.local_shard.grad,unit.layout.shard(expected,axis.rank))
        opt.step();oracle.step()
        restored=unit.full_state_dict()
        for n,p in ref.named_parameters(): close(restored[n],p)
        for key in ('exp_avg','exp_avg_sq'):
            expected=unit.layout.pack([(n,oracle.state[p][key]) for n,p in ref.named_parameters()])
            close(opt.state[unit.local_shard][key],unit.layout.shard(expected,axis.rank))


def moe(axis,device):
    from megatron_mini.moe.layer import ExpertParallelMoE,synchronize_moe_gradients
    E=4;k=2
    if E%axis.size: raise ValueError('MoE lab EP must divide 4')
    router=nn.Linear(8,E).to(device)
    experts=nn.ModuleList([nn.Sequential(nn.Linear(8,12),nn.GELU(),nn.Linear(12,8)) for _ in range(E)]).to(device)
    local_experts={i:copy.deepcopy(experts[i]) for i in range(axis.rank*(E//axis.size),(axis.rank+1)*(E//axis.size))}
    module=ExpertParallelMoE(copy.deepcopy(router),local_experts,E,k,axis)
    x,_=_input(axis.rank,device);all_x=torch.cat([_input(r,device)[0] for r in range(axis.size)])
    probs=router(all_x).softmax(-1);scores,indices=probs.topk(k,dim=-1)
    gates=scores/scores.sum(-1,keepdim=True)
    rows=[]
    for token in range(all_x.shape[0]):
        rows.append(sum(gates[token,j]*experts[int(indices[token,j])](all_x[token]) for j in range(k)))
    ref=torch.stack(rows)
    out,aux=module(x);close(out,ref.chunk(axis.size,0)[axis.rank]);assert torch.isfinite(aux)
    out.square().mean().backward();ref.square().mean().backward()
    synchronize_moe_gradients(list(module.router.parameters()),list(module.experts.parameters()),axis,Axis(),axis.size)
    for p,q in zip(module.router.parameters(),router.parameters()): close(p.grad,q.grad)
    for name,expert in module.experts.items():
        for p,q in zip(expert.parameters(),experts[int(name)].parameters()):
            close(p.grad,torch.zeros_like(q) if q.grad is None else q.grad)
    # aux loss has a separate differentiable router path.
    module.zero_grad(set_to_none=True);_,aux=module(x);aux.backward()
    assert module.router.weight.grad is not None and torch.isfinite(module.router.weight.grad).all()


def precision(axis,device):
    from megatron_mini.runtime.precision import autocast_context,unscale_and_check
    model=nn.Linear(8,4).to(device);x,_=_input(axis.rank,device)
    with autocast_context(device.type,'bf16'): out=model(x)
    assert out.dtype==torch.bfloat16
    out.float().square().mean().backward()
    assert all(p.grad.dtype==torch.float32 for p in model.parameters())
    p=nn.Parameter(torch.ones(3,device=device));p.grad=torch.full_like(p,16.0)
    assert not unscale_and_check([p],8.0,[axis]);close(p.grad,torch.full_like(p,2.0))
    p.grad=torch.ones_like(p)
    if axis.rank==axis.size-1: p.grad[0]=float('inf')
    assert unscale_and_check([p],8.0,[axis])


def roles(axis,device):
    from megatron_mini.runtime.grad_roles import GradRole,Reduction,finalize_gradients
    p=nn.Parameter(torch.zeros(2,device=device));p.grad=torch.full_like(p,axis.rank+1.)
    expected=(axis.size+1)/2
    rule=GradRole((Reduction(axis,axis.size),),True,'dense')
    norm=finalize_gradients([('p',p)],{'p':rule},[],max_norm=0.25)
    close(torch.as_tensor(norm,device=device),torch.tensor(2**0.5*expected,device=device))
    assert abs(p.grad.norm().item()-0.25)<1e-5


def model(axis,device):
    from megatron_mini.model.llama import rms_norm,apply_rope,gqa_attention,LlamaBlock
    x=torch.randn(2,5,8,device=device,dtype=torch.float64,requires_grad=True)
    w=torch.randn(8,device=device,dtype=torch.float64,requires_grad=True)
    y=rms_norm(x,w);expected=x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()*w
    close(y,expected)
    actual_grad=torch.autograd.grad(y.sum(),(x,w),retain_graph=True)
    ref_grad=torch.autograd.grad(expected.sum(),(x,w))
    for a,b in zip(actual_grad,ref_grad): close(a,b)
    z=torch.randn(2,4,5,8,device=device,requires_grad=True)
    close(apply_rope(z,torch.zeros(5,device=device)),z)
    rotated=apply_rope(z,torch.arange(5,device=device)+11)
    close(rotated.square().sum(-1),z.square().sum(-1))
    q=torch.randn(1,4,5,4,device=device,requires_grad=True)
    k=torch.randn(1,2,5,4,device=device,requires_grad=True);v=torch.randn_like(k,requires_grad=True)
    from megatron_mini.model.dense import causal_attention
    ref=torch.cat([causal_attention(q[:,h:h+1],k[:,h//2:h//2+1],v[:,h//2:h//2+1]) for h in range(4)],1)
    out=gqa_attention(q,k,v);close(out,ref)
    a=torch.autograd.grad(out.square().sum(),(q,k,v),retain_graph=True)
    b=torch.autograd.grad(ref.square().sum(),(q,k,v))
    for a,b in zip(a,b): close(a,b)
    block=LlamaBlock().to(device);block(torch.randn(2,8,32,device=device)).sum().backward()
    assert all(p.grad is not None for p in block.parameters())


def text(axis,device):
    from megatron_mini.runtime.text import masked_token_loss
    n=axis.rank+1
    logits=torch.arange(24,device=device,dtype=torch.float32).reshape(1,6,4).clone().requires_grad_()
    targets=torch.tensor([[0,1,2,3,0,1]],device=device)
    mask=torch.arange(6,device=device).view(1,6)<min(n,6)
    stats=masked_token_loss(logits,targets,mask,axis)
    total=sum(min(r+1,6) for r in range(axis.size))
    losses=F.cross_entropy(logits.view(-1,4),targets.view(-1),reduction='none')
    expect=(losses*mask.view(-1)).sum()/total
    close(stats.backward_loss,expect);assert stats.valid_tokens==total
    a=torch.autograd.grad(stats.backward_loss,logits,retain_graph=True)[0]
    b=torch.autograd.grad(expect,logits)[0];close(a,b)


class FakeWork:
    def __init__(self,events,name): self.events,self.name=events,name
    def wait(self): self.events.append(('wait',self.name))


def bucket(axis,device):
    from megatron_mini.runtime.overlap import Bucket,Pending,mark_bucket_ready
    events=[];b=Bucket(('a','b'),torch.ones(4,device=device))
    def launch(buf):
        events.append(('launch','bucket'));return Pending(FakeWork(events,'bucket'),(buf,))
    mark_bucket_ready(b,'a',False,launch);mark_bucket_ready(b,'b',False,launch)
    assert not events
    mark_bucket_ready(b,'a',True,launch);assert not events
    mark_bucket_ready(b,'b',True,launch);assert events==[('launch','bucket')]
    assert b.pending is not None
    b.pending.wait();assert b.pending.finished


def overlap(axis,device):
    from megatron_mini.runtime.overlap import Pending,prefetch_next,tp_chunked_overlap,ep_chunked_overlap,cp_ring_overlap
    events=[]
    def launch(i):
        events.append(('launch',i));return Pending(FakeWork(events,i),(torch.tensor(float(i)),))
    def compute(i,x):
        assert ('wait',i) in events;events.append(('compute',i));return x+1
    out=tp_chunked_overlap(3,launch,compute)
    close(torch.stack(out),torch.tensor([1.,2.,3.]))
    assert events.index(('launch',1))<events.index(('compute',0))
    events.clear()
    out=ep_chunked_overlap(3,launch,compute,lambda i,y:y*2)
    close(torch.stack(out),torch.tensor([2.,4.,6.]))
    assert events.index(('launch',1))<events.index(('compute',0))
    events.clear()
    result,pending=prefetch_next(4,1,launch,lambda x:events.append(('compute','current')) or x*2)
    assert result==8 and not pending.finished
    assert events[0]==('launch',1)
    events.clear()
    cp_ring_overlap(3,launch,lambda i:events.append(('consume',i)))
    assert events.index(('launch',0))<events.index(('consume',0))<events.index(('wait',0))
    assert ('launch',2) not in events


def reshard(axis,device):
    from megatron_mini.runtime.checkpoint import reshard_flat_state
    old=[torch.arange(8.,device=device),torch.tensor([8,9,10,11,12,0,0,0.],device=device)]
    out=reshard_flat_state(old,13,4)
    assert len(out)==4 and all(x.numel()==4 for x in out)
    close(torch.cat(out)[:13],torch.arange(13.,device=device))
    close(torch.cat(out)[13:],torch.zeros(3,device=device))
    close(reshard_flat_state(old,13,1)[0],torch.arange(13.,device=device))


def checkpoint(axis,device):
    from megatron_mini.runtime.checkpoint import save_sharded_checkpoint
    if axis.size>1:
        raise ValueError('checkpoint contract uses shared filesystem: run unit test then recipe manifest test')
    with tempfile.TemporaryDirectory() as d:
        payload={'model':{'p':torch.arange(4.)},'optimizer':{},'step':3,'metadata':{},'rng':torch.get_rng_state()}
        manifest=save_sharded_checkpoint(d,0,payload,axis)
        assert Path(manifest).is_file()
        import json
        meta=json.loads(Path(manifest).read_text())
        assert meta['world_size']==1 and meta['step']==3
        assert len(meta['shards'])==1 and all((Path(d)/s).is_file() for s in meta['shards'])


def run_case(case,axis,device):
    fn=globals().get(case)
    if fn is None: raise ValueError(f'no executable case: {case}')
    return fn(axis,device)

# v3 executable mechanism checks; student algorithms remain in their own modules.
from megatron_mini.labs.memory_checks import (residency,zero3,zero3_sac,offload_transfer,
    offload_optimizer,offload_param,offload_activation,sac,offload_checkpoint)
