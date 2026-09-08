"""显式选择才运行：pytest -q --exercises tests/test_exercises.py。"""
import pytest
import torch
from torch.nn import functional as F
from megatron_mini.config import ModelConfig
from megatron_mini.data import make_batch, split_microbatches
from megatron_mini.distributed.topology import ParallelDims, GroupSpec, plan_groups
from megatron_mini.distributed.mesh import Mesh
from megatron_mini.training.loss import causal_lm_loss
from megatron_mini.training.schedules import forward_backward_no_pipeline
from megatron_mini.training.grad import clip_grad_norm
from megatron_mini.model.dense import build_dense
from megatron_mini.model.state import build_stage
from megatron_mini.tensor_parallel.loss import vocab_parallel_cross_entropy
from megatron_mini.checkpoint import save_checkpoint, load_checkpoint

pytestmark = pytest.mark.exercise


def test_token_loss():
    logits = torch.randn(3,5,16,dtype=torch.float64,requires_grad=True)
    targets = torch.randint(0,16,(3,5))
    loss = causal_lm_loss(logits,targets)
    expected = F.cross_entropy(logits.reshape(-1,16),targets.reshape(-1))
    torch.testing.assert_close(loss,expected)
    g1, = torch.autograd.grad(loss,logits,retain_graph=True)
    g2, = torch.autograd.grad(expected,logits)
    torch.testing.assert_close(g1,g2)


def test_group_plan():
    dims = ParallelDims(2,2,2)
    plan = plan_groups(dims)
    expected = [GroupSpec("tp",x) for x in ((0,1),(2,3),(4,5),(6,7))]
    expected += [GroupSpec("pp",x) for x in ((0,2),(1,3),(4,6),(5,7))]
    expected += [GroupSpec("dp",x) for x in ((0,4),(1,5),(2,6),(3,7))]
    assert plan == expected
    for dims in [ParallelDims(),ParallelDims(2,3,2),ParallelDims(1,2,2)]:
        plan = plan_groups(dims)
        for axis in ("tp","pp","dp"):
            covered = [r for g in plan if g.axis==axis for r in g.ranks]
            assert sorted(covered) == list(range(dims.world_size))
            assert all(len(g.ranks)==getattr(dims,axis) for g in plan if g.axis==axis)


def test_single_rank_vocab_loss():
    x = (torch.randn(2,3,16,dtype=torch.float64)*800).requires_grad_()
    target = torch.tensor([[0,7,15],[8,1,14]])
    actual = vocab_parallel_cross_entropy(x,target,Mesh.local())
    expected = F.cross_entropy(x.reshape(-1,16),target.reshape(-1),reduction="none").reshape(2,3)
    torch.testing.assert_close(actual,expected)
    # 任意 upstream grad，避免只支持 mean 的假实现。
    grad = torch.randn_like(actual)
    g1, = torch.autograd.grad(actual,x,grad,retain_graph=True)
    g2, = torch.autograd.grad(expected,x,grad)
    torch.testing.assert_close(g1,g2)


def test_student_model_single_rank_forward_and_gradients():
    cfg=ModelConfig(seq_len=8)
    dense=build_dense(cfg); student=build_stage(cfg,Mesh.local())
    batch=make_batch(cfg,0,4)
    actual=student(batch.input_ids); expected=dense(batch.input_ids)
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
    causal_lm_loss(actual,batch.target_ids).backward()
    F.cross_entropy(expected.reshape(-1,cfg.vocab_size),batch.target_ids.reshape(-1)).backward()
    full=dict(dense.named_parameters())
    for name,p in student.named_parameters():
        torch.testing.assert_close(p.grad,full[name].grad,rtol=3e-5,atol=3e-6)


def test_student_gradient_accumulation():
    cfg=ModelConfig(seq_len=8)
    dense=build_dense(cfg); student=build_stage(cfg,Mesh.local())
    batch=make_batch(cfg,1,6)
    expected=F.cross_entropy(dense(batch.input_ids).reshape(-1,cfg.vocab_size),batch.target_ids.reshape(-1))
    expected.backward()
    actual=forward_backward_no_pipeline(student,split_microbatches(batch,2))
    torch.testing.assert_close(actual,expected.detach(),rtol=3e-5,atol=3e-6)
    full=dict(dense.named_parameters())
    for name,p in student.named_parameters():
        torch.testing.assert_close(p.grad,full[name].grad,rtol=3e-5,atol=3e-6)


def test_single_rank_clip_norm():
    cfg=ModelConfig(seq_len=8)
    dense=build_dense(cfg); student=build_stage(cfg,Mesh.local())
    for name,p in student.named_parameters():
        p.grad=torch.ones_like(p)*0.1
    for p in dense.parameters(): p.grad=torch.ones_like(p)*0.1
    expected=torch.nn.utils.clip_grad_norm_(dense.parameters(),0.25)
    actual=clip_grad_norm(student,student.mesh,0.25)
    torch.testing.assert_close(actual.to(expected.dtype),expected,rtol=3e-5,atol=3e-6)
    for p,q in zip(student.parameters(),dense.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=3e-6)


def test_checkpoint_optimizer_and_rng(tmp_path):
    cfg=ModelConfig(seq_len=8)
    a=build_stage(cfg,Mesh.local()); opt=torch.optim.AdamW(a.parameters())
    # 不依赖尚未完成的 student forward，使 T19/T20 可以独立测试。
    for p in a.parameters(): p.grad=torch.ones_like(p)
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.manual_seed(777)
    save_checkpoint(tmp_path,a,opt,7)
    wanted=torch.rand(5)
    b=build_stage(cfg,Mesh.local()); restored=torch.optim.AdamW(b.parameters())
    step=load_checkpoint(tmp_path,b,restored)
    assert step==7
    torch.testing.assert_close(torch.rand(5),wanted,rtol=0,atol=0)
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        for key in ("step","exp_avg","exp_avg_sq"):
            torch.testing.assert_close(opt.state[p][key],restored.state[q][key],rtol=0,atol=0)
