"""Numerical tests for memory exercises. No fallback solutions live in this module."""
import copy
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


def close(a,b):
    torch.testing.assert_close(a.detach().cpu(),b.detach().cpu(),rtol=4e-5,atol=4e-6)


def residency(axis,device):
    from megatron_mini.memory.residency import ShardHandle,Residency
    local=torch.arange(5.,device=device)+axis.rank*5
    h=ShardHandle(local,axis)
    out=h.materialize()
    close(out,torch.arange(5.*axis.size,device=device))
    assert h.state is Residency.FULL and h.stats.gathers==1
    assert h.stats.live_full_bytes==out.numel()*out.element_size()
    try: h.materialize()
    except RuntimeError: pass
    else: raise AssertionError('double materialization must be rejected')
    del out
    h.release()
    assert h.full is None and h.state is Residency.SHARDED
    assert h.stats.live_full_bytes==0 and local.numel()==5
    h.materialize();h.release()
    assert h.stats.gathers==h.stats.releases==2


class CountAxis:
    def __init__(self,axis):
        self.base=axis;self.gathers=0;self.reductions=0
        self.size,self.rank,self.group=axis.size,axis.rank,axis.group
    def gather(self,*a,**kw):
        self.gathers+=1;return self.base.gather(*a,**kw)
    def sum(self,*a,**kw): return self.base.sum(*a,**kw)
    def reduce_scatter(self,*a,**kw):
        self.reductions+=1;return self.base.reduce_scatter(*a,**kw)


def _zero3(axis,device,sac=False):
    from megatron_mini.optimizer.zero3 import build_zero3_sequential
    axis=CountAxis(axis)
    source=nn.Sequential(nn.Sequential(nn.Linear(8,12),nn.GELU()),
                         nn.Sequential(nn.Linear(12,4),nn.Sigmoid())).to(device)
    ref=copy.deepcopy(source)
    model=build_zero3_sequential(source,axis,sac=sac)
    del source
    assert isinstance(model,nn.Sequential) and len(model)==2
    assert set(dict(model.named_parameters()))=={'0.local_shard','1.local_shard'}
    assert all(p.is_meta for u in model for p in u.template.parameters())
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    oracle=torch.optim.AdamW(ref.parameters(),lr=.003)
    M=2
    for step in range(3):
        opt.zero_grad(set_to_none=True);oracle.zero_grad(set_to_none=True)
        def data(rank,m):
            return (torch.arange(16.,device=device).reshape(2,8)/16+rank*.1+m*.05+step*.01).requires_grad_()
        xs=[data(axis.rank,m) for m in range(M)]
        ref_xs=[[data(r,m) for m in range(M)] for r in range(axis.size)]
        before=(axis.gathers,axis.reductions)
        # Two live graphs detect accidentally shared SAC caches across microbatches.
        outs=[model(x) for x in xs]
        for x,out in zip(xs,outs): close(out,ref(x))
        for out in reversed(outs): (out.square().mean()/M).backward()
        expected_loss=sum(ref(x).square().mean() for row in ref_xs for x in row)/(axis.size*M)
        expected_loss.backward()
        assert axis.gathers-before[0]==2*M*len(model), 'AG repeated inside SAC or skipped at backward'
        assert axis.reductions-before[1]==M*len(model), 'wrong RS lifecycle'
        for x,q in zip(xs,ref_xs[axis.rank]): close(x.grad,q.grad*axis.size)
        for u,block in zip(model,ref):
            packed=u.layout.pack([(n,p.grad) for n,p in block.named_parameters()])
            close(u.local_shard.grad,u.layout.shard(packed,axis.rank))
        # Validate clipping the actual optimizer-owned shard, with true global norm.
        norm=axis.sum(sum((p.grad.double().square().sum() for p in model.parameters()),
                          torch.zeros((),device=device,dtype=torch.float64))).sqrt()
        refnorm=sum((p.grad.double().square().sum() for p in ref.parameters()),
                    torch.zeros((),device=device,dtype=torch.float64)).sqrt()
        close(norm,refnorm)
        factor=min(1.,.025/(norm.item()+1e-6))
        for p in model.parameters(): p.grad.mul_(factor)
        for p in ref.parameters(): p.grad.mul_(factor)
        opt.step();oracle.step()
        for u,block in zip(model,ref):
            restored=u.full_state_dict()
            for n,p in block.named_parameters(): close(restored[n],p)
            for key in ('exp_avg','exp_avg_sq'):
                packed=u.layout.pack([(n,oracle.state[p][key]) for n,p in block.named_parameters()])
                close(opt.state[u.local_shard][key],u.layout.shard(packed,axis.rank))
    if sac:
        for u in model:
            assert u.stats.forward['aten.addmm.default']>0
            assert u.stats.recompute['aten.addmm.default']==0, 'cached GEMMs actually ran again'


def zero3(axis,device): _zero3(axis,device)
def zero3_sac(axis,device): _zero3(axis,device,True)


def offload_transfer(axis,device):
    from megatron_mini.offload.transfer import submit_copy
    src=torch.arange(30.).reshape(5,6).t() # noncontiguous source
    if device.type=='cuda': src=src.contiguous().pin_memory()
    stream=torch.cuda.Stream(device=device) if device.type=='cuda' else None
    upload=submit_copy(src,device,pin_memory=device.type=='cuda',stream=stream)
    if stream is not None:
        value=upload.wait_stream(torch.cuda.current_stream(device))
    else: value=upload.wait_host()
    doubled=value*2
    download=submit_copy(doubled,'cpu',pin_memory=device.type=='cuda',stream=stream)
    close(download.wait_host(),src*2)
    copied=upload.wait_host().cpu().clone()
    src.add_(100)
    close(upload.tensor,copied)
    assert not upload.keepalive and not download.keepalive


def _equal_adam(a,b):
    close(a['master'],b['master'])
    for key in ('step','exp_avg','exp_avg_sq'):
        close(a['optimizer']['state'][0][key],b['optimizer']['state'][0][key])


def offload_optimizer(axis,device):
    from megatron_mini.offload.optimizer import CPUShardAdam
    init=torch.arange(7.)/13+axis.rank*.05
    student=CPUShardAdam(init.to(device),lr=.002)
    ref=nn.Parameter(init.clone());oracle=torch.optim.AdamW([ref],lr=.002)
    for step in range(3):
        grad=(init+step*.2).cos()
        student.step(grad.to(device))
        ref.grad=grad.clone();oracle.step()
        close(student.master,ref)
        assert student.master.device.type=='cpu'
        for key in ('step','exp_avg','exp_avg_sq'):
            assert student.optim.state[student.master][key].device.type=='cpu'
            close(student.optim.state[student.master][key],oracle.state[ref][key])
    before=student.state_dict()
    student.step(torch.full_like(init,float('inf')).to(device),found_inf=True)
    _equal_adam(before,student.state_dict())
    restored=CPUShardAdam(torch.zeros_like(init),lr=.002);restored.load_state_dict(before)
    grad=init.sin().to(device)
    student.step(grad);restored.step(grad)
    _equal_adam(student.state_dict(),restored.state_dict())


def offload_param(axis,device):
    from megatron_mini.offload.parameters import CPUParamStore
    owner=nn.Parameter(torch.arange(5.)+5*axis.rank)
    if device.type=='cuda': owner=nn.Parameter(owner.detach().pin_memory())
    store=CPUParamStore(owner,axis,device,pin_memory=device.type=='cuda')
    assert store.cpu_shard is owner
    close(store.materialize(),torch.arange(5.*axis.size))
    assert store.full.device==device
    try: store.materialize()
    except RuntimeError: pass
    else: raise AssertionError('must not overwrite a live materialization')
    store.release()
    assert store.full is None and store.staging is None and store.transfer is None
    with torch.no_grad(): owner.add_(7)
    close(store.materialize(),torch.arange(5.*axis.size)+7)
    store.release()
    assert store.stats.gathers==store.stats.releases==2


def offload_activation(axis,device):
    from megatron_mini.offload.activations import ActivationOffloader
    model=nn.Sequential(nn.Linear(8,12),nn.GELU(),nn.Linear(12,4)).to(device)
    oracle=copy.deepcopy(model)
    x=torch.randn(3,8,device=device,requires_grad=True);z=x.detach().clone().requires_grad_()
    hook=ActivationOffloader(model.parameters(),pin_memory=device.type=='cuda',emulate_cpu=device.type=='cpu')
    with hook.hooks(): out=model(x).sin()
    assert hook.records and hook.offloaded_bytes>0
    assert all(not r.transfer.keepalive and r.transfer.tensor.device.type=='cpu' for r in hook.records)
    out.square().sum().backward();expected=oracle(z).sin();expected.square().sum().backward()
    close(out,expected);close(x.grad,z.grad)
    for p,q in zip(model.parameters(),oracle.parameters()): close(p.grad,q.grad)
    count=len(hook.records)
    hook.pack(model[0].weight.t()) # protected storage, even though view is not a Parameter
    hook.pack(torch.tensor([1,2],device=device))
    assert len(hook.records)==count
    # A saved non-contiguous activation must be a snapshot, not a live alias.
    value=torch.arange(24.,device=device).reshape(4,6).t()
    expected=value.clone();packed=hook.pack(value);hook.drain();value.add_(10)
    close(hook.unpack(packed),expected)


def sac(axis,device):
    from megatron_mini.runtime.sac import SACStats,save_matmuls_policy,selective_checkpoint
    from torch.utils.checkpoint import CheckpointPolicy,SelectiveCheckpointContext
    ctx=SelectiveCheckpointContext(is_recompute=False)
    for op in (torch.ops.aten.mm.default,torch.ops.aten.addmm.default,torch.ops.aten.bmm.default):
        assert save_matmuls_policy(ctx,op)==CheckpointPolicy.MUST_SAVE
    assert save_matmuls_policy(ctx,torch.ops.aten.sin.default)==CheckpointPolicy.PREFER_RECOMPUTE
    model=nn.Sequential(nn.Linear(8,12),nn.Dropout(.25),nn.GELU(),nn.Linear(12,4),nn.Sigmoid()).to(device)
    oracle=copy.deepcopy(model)
    xs=[torch.randn(3,8,device=device,requires_grad=True) for _ in range(2)]
    ref_xs=[x.detach().clone().requires_grad_() for x in xs]
    stats=SACStats()
    torch.manual_seed(789)
    outputs=[selective_checkpoint(model,x,stats=stats) for x in xs]
    for y in reversed(outputs): y.square().sum().backward()
    rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
    torch.manual_seed(789)
    expected=[oracle(x) for x in ref_xs]
    for y in reversed(expected): y.square().sum().backward()
    assert torch.equal(rng,torch.get_rng_state())
    if cuda_rng is not None: assert torch.equal(cuda_rng,torch.cuda.get_rng_state(device))
    for y,z,x,rx in zip(outputs,expected,xs,ref_xs): close(y,z);close(x.grad,rx.grad)
    for p,q in zip(model.parameters(),oracle.parameters()): close(p.grad,q.grad)
    assert stats.forward['aten.addmm.default']==4
    assert stats.recompute['aten.addmm.default']==0
    assert sum(stats.recompute.values())>0, 'no recomputation was observed at all'


def offload_checkpoint(axis,device):
    from megatron_mini.offload.optimizer import CPUShardAdam
    from megatron_mini.offload.checkpoint import snapshot_offloaded
    from megatron_mini.offload.transfer import Transfer
    owner=CPUShardAdam(torch.arange(5.).to(device)/7,lr=.004)
    owner.step(torch.ones(5,device=device))
    calls=[]
    class Event:
        def synchronize(self): calls.append('wait')
    handle=Transfer(torch.ones(1),Event(),(torch.ones(1),))
    metadata={'rank':axis.rank,'cpu_rng':torch.get_rng_state(),'placement':'cpu','cursor':1}
    snapshot=snapshot_offloaded([owner],[handle],metadata,next_step=1)
    assert calls==['wait'] and snapshot['format_version']==1 and snapshot['next_step']==1
    restored=CPUShardAdam(torch.zeros(5),lr=.004);restored.load_state_dict(snapshot['owners'][0])
    old=snapshot['owners'][0]['master'].clone()
    grad=torch.arange(5.,device=device).sin()
    owner.step(grad);restored.step(grad)
    _equal_adam(owner.state_dict(),restored.state_dict())
    close(snapshot['owners'][0]['master'],old)
    metadata['cpu_rng'].zero_()
    assert snapshot['metadata']['cpu_rng'].any()
