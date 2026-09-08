"""Ready plumbing only; these assertions never claim sharding/offload/SAC implemented."""
import pytest
import torch
from megatron_mini.offload.config import OffloadConfig
from megatron_mini.offload.transfer import Transfer
from megatron_mini.runtime.config import RecipeConfig
from megatron_mini.labs.catalog import RECIPES,recipe_tasks
from megatron_mini.runtime.sac import observed_contexts,SACStats
from torch.utils.checkpoint import CheckpointPolicy

@pytest.mark.parametrize('sharding',['zero3','fsdp','torch_fsdp2'])
def test_full_shard_configs_are_explicit(sharding):
    c=RecipeConfig(dp=2,sharding=sharding,offload=OffloadConfig(optimizer='cpu',parameters='cpu'))
    assert c.validate()['sharding']==sharding

@pytest.mark.parametrize('sharding,offload',[
 ('zero2',OffloadConfig(parameters='cpu',optimizer='cpu')),
 ('zero3',OffloadConfig(parameters='cpu')),
 ('zero3',OffloadConfig(parameters='nvme')),
 ('ddp',OffloadConfig(optimizer='cpu')),
 ('torch_fsdp2',OffloadConfig(optimizer='cpu')),
 ('zero3',OffloadConfig(optimizer='cpu',async_transfer=True)),
 ('ddp',OffloadConfig(pin_memory=True,async_transfer=True)),
])
def test_invalid_placements_rejected(sharding,offload):
    with pytest.raises(ValueError): RecipeConfig(sharding=sharding,offload=offload).validate()


def test_memory_recipe_tasks_are_not_silent_flags():
    c=RECIPES['zero3_cpu_sac'];c.validate();ids=set(recipe_tasks(c))
    assert {'T55','T61','T62','T63','T64','T65','T66','T67','T68','T69','T70','T71','T72'}<=ids
    assert 'T33' not in ids # no redundant whole-block AC auto-added to SAC


def test_transfer_host_and_stream_wait_differ():
    calls=[]
    class E:
        def synchronize(self): calls.append('host')
    class S:
        def wait_event(self,e): calls.append('stream')
    tensor=torch.ones(3)
    h=Transfer(tensor,E(),(tensor,))
    assert h.wait_stream(S()) is tensor
    assert not h.host_ready and h.keepalive and calls==['stream']
    h.wait_host();h.wait_host()
    assert h.host_ready and not h.keepalive and calls==['stream','host']


def test_executed_op_observer_is_below_cache():
    # Instrumentation check using an explicit policy, not the T67/T68 student path.
    stats=SACStats()
    def policy(ctx,op,*args,**kwargs):
        return CheckpointPolicy.MUST_SAVE if op==torch.ops.aten.mm.default else CheckpointPolicy.PREFER_RECOMPUTE
    fwd,replay=observed_contexts(policy,stats)
    x=torch.randn(3,4);w=torch.randn(4,5)
    with fwd: result=torch.mm(x,w).sin()
    with replay: other=torch.mm(x,w).sin()
    torch.testing.assert_close(result,other)
    assert stats.forward['aten.mm.default']==1 and stats.recompute['aten.mm.default']==0
    assert stats.recompute['aten.sin.default']==1
