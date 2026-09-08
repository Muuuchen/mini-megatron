"""Explicit --exercises enables incomplete algorithms; default skips are NOT feature passes."""
import pytest
import torch
from megatron_mini.distributed.axis import Axis
from megatron_mini.labs.checks import run_case

pytestmark=pytest.mark.exercise

@pytest.mark.parametrize('case',['topology','sp','cp','pipeline','recompute','flat','zero1',
                               'zero2','fsdp','moe','precision','roles','model','text',
                               'bucket','overlap','reshard','checkpoint'])
def test_v2_local_case(case):
    run_case(case,Axis(),torch.device('cpu'))


def test_online_update_masked_and_partitioned():
    from megatron_mini.context_parallel.ring import online_update
    scores=torch.randn(1,2,4,6)
    v=torch.randn(1,2,6,3)
    m=torch.full((1,2,4,1),float('-inf'));l=torch.zeros_like(m);o=torch.zeros(1,2,4,3)
    m,l,o=online_update(m,l,o,scores[...,:2],v[...,:2,:],torch.zeros_like(scores[...,:2],dtype=torch.bool))
    assert not torch.isnan(m).any() and not torch.isnan(o).any()
    assert (l==0).all() and (o==0).all()
    for begin,end in ((0,2),(2,6)):
        m,l,o=online_update(m,l,o,scores[...,begin:end],v[...,begin:end,:],True)
    torch.testing.assert_close(o/l,scores.softmax(-1)@v,rtol=3e-5,atol=3e-6)

@pytest.mark.parametrize('p,m',[(2,1),(2,4),(4,1),(4,8)])
def test_1f1b_events(p,m):
    from megatron_mini.pipeline_parallel.planner import plan_1f1b,validate_schedule_simulation
    peak=validate_schedule_simulation([plan_1f1b(r,p,m) for r in range(p)],p,m)
    assert all(v<=min(m,p-r) for r,v in enumerate(peak))


def test_interleaved_events():
    from megatron_mini.pipeline_parallel.planner import plan_interleaved,validate_schedule_simulation
    plans=[plan_interleaved(r,2,4,2) for r in range(2)]
    validate_schedule_simulation(plans,2,4,2)
