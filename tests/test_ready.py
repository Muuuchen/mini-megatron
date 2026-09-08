"""这些是已完成脚手架的测试，不代表学生的分布式实现已通过。"""
import pytest
import torch
from torch.nn import functional as F
from megatron_mini.config import ModelConfig
from megatron_mini.data import make_batch, split_microbatches
from megatron_mini.distributed.topology import ParallelDims
from megatron_mini.distributed.mesh import Mesh
from megatron_mini.model.dense import build_dense
from megatron_mini.model.parallel import GPTStage
from megatron_mini.model.state import load_dense_state, shard_like
from megatron_mini.checkpoint import metadata, validate_metadata
from megatron_mini.tasks import TASKS, pending_tasks


@pytest.mark.parametrize("dims", [ParallelDims(), ParallelDims(2,2,2), ParallelDims(2,3,4)])
def test_rank_roundtrip(dims):
    for rank in range(dims.world_size):
        assert dims.rank(*dims.coords(rank)) == rank


@pytest.mark.parametrize("kwargs,tp,pp", [({"num_heads":3},1,1), ({"vocab_size":63},2,1),
                                            ({"num_layers":3},1,2), ({"seq_len":0},1,1)])
def test_config_rejects_invalid(kwargs, tp, pp):
    with pytest.raises(ValueError): ModelConfig(**kwargs).validate(tp, pp)


def test_dp_data_matches_global_slice():
    cfg = ModelConfig()
    global_batch = make_batch(cfg, 3, 12)
    pieces = [make_batch(cfg, 3, 4, d, 3) for d in range(3)]
    torch.testing.assert_close(torch.cat([b.input_ids for b in pieces]), global_batch.input_ids)
    torch.testing.assert_close(torch.cat([b.target_ids for b in pieces]), global_batch.target_ids)
    assert torch.equal(global_batch.target_ids[:, :-1], global_batch.input_ids[:, 1:])


def test_microbatch_roundtrip():
    batch = make_batch(ModelConfig(), 0, 6)
    micro = split_microbatches(batch, 2)
    assert len(micro) == 3
    torch.testing.assert_close(torch.cat([b.input_ids for b in micro]), batch.input_ids)
    with pytest.raises(ValueError): split_microbatches(batch, 4)


def test_dense_deterministic_initialization():
    a, b = build_dense(ModelConfig()), build_dense(ModelConfig())
    for x, y in zip(a.parameters(), b.parameters()): torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_dense_causality():
    cfg = ModelConfig()
    model = build_dense(cfg)
    x = make_batch(cfg, 0, 2).input_ids
    y = x.clone(); y[:, 8:] = (y[:, 8:] + 9) % cfg.vocab_size
    torch.testing.assert_close(model(x)[:, :8], model(y)[:, :8], rtol=0, atol=0)


def test_dense_grad_accumulation_oracle():
    cfg = ModelConfig(seq_len=8)
    a, b = build_dense(cfg), build_dense(cfg)
    batch = make_batch(cfg, 0, 6)
    F.cross_entropy(a(batch.input_ids).reshape(-1,cfg.vocab_size), batch.target_ids.reshape(-1)).backward()
    for mb in split_microbatches(batch, 2):
        (F.cross_entropy(b(mb.input_ids).reshape(-1,cfg.vocab_size), mb.target_ids.reshape(-1))/3).backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=3e-7)


def test_dense_can_learn_synthetic_task():
    cfg = ModelConfig(seq_len=8, num_layers=2)
    model = build_dense(cfg)
    batch = make_batch(cfg, 0, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    losses = []
    for _ in range(16):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch.input_ids).reshape(-1,cfg.vocab_size),batch.target_ids.reshape(-1))
        losses.append(loss.item()); loss.backward(); opt.step()
    assert losses[-1] < losses[0] * 0.6


def test_shard_initialization_and_stage_ownership():
    # 仅测试构造/切片，不启动分布式或调用任何 TODO。
    cfg = ModelConfig()
    dense = build_dense(cfg).state_dict()
    dims = ParallelDims(tp=2,pp=2,dp=1)
    for rank in range(4):
        mesh = Mesh(dims,rank,torch.device("cpu"))
        stage = GPTStage(cfg,mesh)
        load_dense_state(stage,dense)
        assert hasattr(stage,"tok") == mesh.first
        assert hasattr(stage,"head") == mesh.last
        assert len(stage.blocks) == 2
        for name,p in stage.named_parameters():
            torch.testing.assert_close(p,shard_like(dense[name],p,mesh),rtol=0,atol=0)


def test_checkpoint_metadata_rejects_other_topology():
    stage = GPTStage(ModelConfig(), Mesh.local())
    good = metadata(stage)
    validate_metadata(good,stage)
    bad = dict(good); bad["rank"] = 1
    with pytest.raises(ValueError): validate_metadata(bad,stage)
    stage.run_config = {"micro_batch": 2, "microbatches": 4}
    with pytest.raises(ValueError): validate_metadata(good,stage)


def test_task_manifest_covers_remaining_placeholders():
    assert len(TASKS) == 72
    assert set(pending_tasks()) <= TASKS.keys()
