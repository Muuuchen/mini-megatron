"""These tests validate delivered plumbing, never substitute for student feature tests."""
import ast
from pathlib import Path
import pytest
import torch
from megatron_mini.distributed.axis import Axis
from megatron_mini.distributed.topology_v2 import MeshDims
from megatron_mini.runtime.config import RecipeConfig
from megatron_mini.runtime.precision import DynamicScale
from megatron_mini.runtime.overlap import Pending
from megatron_mini.runtime.text import byte_examples
from megatron_mini.pipeline_parallel.planner import Event,validate_schedule_simulation
from megatron_mini.training.engine_v2 import TrainRuntime,train_step_v2
from megatron_mini.labs.catalog import CASES,RECIPES,recipe_tasks
from megatron_mini.tasks import TASKS,pending_tasks

@pytest.mark.parametrize('dims',[MeshDims(),MeshDims(2,2,4,2,2),MeshDims(1,4,4,2,4)])
def test_v2_rank_roundtrip(dims):
    for r in range(dims.world_size): assert dims.rank(*dims.coords(r))==r
    assert dims.world_size==dims.tp*dims.pp*dims.cp*dims.dp
    assert dims.edp*dims.ep==dims.dp

@pytest.mark.parametrize('kwargs',[{'dp':3,'ep':2},{'tp':0},{'ep':0}])
def test_v2_bad_dims(kwargs):
    with pytest.raises(ValueError): MeshDims(**kwargs)

@pytest.mark.parametrize('kwargs',[{'precision':'fp8'},{'sp':True},{'sharding':'zero4'},
                                  {'tp':2,'dp':2,'ep':2,'model':'moe'},
                                  {'schedule':'interleaved','pp':2}])
def test_v2_reject_invalid_recipe(kwargs):
    with pytest.raises(ValueError): RecipeConfig(**kwargs).validate()

def test_all_recipe_fields_validate():
    for c in RECIPES.values():
        c.validate();assert set(recipe_tasks(c))<=TASKS.keys()

def test_catalog_has_real_case_functions():
    from megatron_mini.labs import checks
    for name,ids in CASES.items():
        assert callable(getattr(checks,name))
        assert set(ids)<=TASKS.keys()

def test_task_sources_and_dependency_ids():
    root=Path(__file__).parents[1]/'megatron_mini'
    assert set(pending_tasks())<=TASKS.keys()
    for task,(day,title,file,deps) in TASKS.items():
        assert (root/file).is_file()
        assert set(deps)<=TASKS.keys()
    # Manifest is acyclic; don't accidentally create impossible exercise prerequisites.
    def visit(t,path):
        assert t not in path
        for d in TASKS[t][3]: visit(d,path|{t})
    for t in TASKS: visit(t,set())

def test_axis_local_equivalence():
    x=torch.arange(12.).reshape(2,6)
    a=Axis();torch.testing.assert_close(a.gather(x,1),x)
    torch.testing.assert_close(a.reduce_scatter(x,1,2),x/2)

def test_byte_sample_shift_and_mask():
    examples=byte_examples('真实文本 samples',7)
    assert examples
    for x,y,m in examples:
        assert x.shape==y.shape==m.shape==(7,)
        n=int(m.sum());assert torch.equal(x[1:n],y[:n-1])
        assert (x[n:]==256).all()

def test_dynamic_scale_persistence():
    s=DynamicScale(16.,2);s.update(False);s.update(False);assert s.value==32.
    s.update(True);assert s.value==16. and s.good_steps==0
    q=DynamicScale();q.load_state_dict(s.state_dict());assert q==s

def test_pending_wait_once_keeps_buffers():
    calls=[]
    class W:
        def wait(self): calls.append(1)
    t=torch.ones(4);h=Pending(W(),(t,));h.wait();h.wait()
    assert calls==[1] and h.buffers[0] is t

def test_schedule_simulator_not_fake_transport():
    plans=[[Event('F',0),Event('B',0)],[Event('F',0),Event('B',0)]]
    assert validate_schedule_simulation(plans,2,1)==[1,1]
    with pytest.raises(AssertionError):
        validate_schedule_simulation([[Event('B',0),Event('F',0)],plans[1]],2,1)

@pytest.mark.parametrize('overflow',[False,True])
def test_v2_engine_order_and_global_skip(overflow):
    log=[]
    def mark(s,value=None):
        def call(*args): log.append(s);return value
        return call
    rt=TrainRuntime(mark('zero'),mark('fb',1.),mark('wait'),mark('unscale',overflow),
                    mark('finalize',0.5),mark('step'),mark('scale'))
    out=train_step_v2(rt,[1,2])
    assert log==(['zero','fb','wait','unscale','scale'] if overflow else
                 ['zero','fb','wait','unscale','finalize','step','scale'])
    assert out['skipped']==overflow
