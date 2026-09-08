"""从仓库根目录用 torchrun -m tools.check_distributed 启动；这是测试，不是训练库。"""
import argparse
import tempfile
import torch
import torch.distributed as dist
from torch.nn import functional as F
from megatron_mini.config import ModelConfig
from megatron_mini.data import make_batch, split_microbatches
from megatron_mini.distributed.topology import ParallelDims
from megatron_mini.distributed.mesh import initialize, shutdown, sum_
from megatron_mini.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from megatron_mini.tensor_parallel.loss import vocab_parallel_cross_entropy, language_model_loss
from megatron_mini.model.dense import build_dense
from megatron_mini.model.state import build_stage, shard_like
from megatron_mini.training.engine import train_step
from megatron_mini.checkpoint import save_checkpoint, load_checkpoint
from megatron_mini.tasks import require_tasks, training_tasks, model_tasks


class Verifier:
    """先完成所有 collective，再让所有 rank 一起报告失败；减少测试自身造成的挂起。"""
    def __init__(self, mesh):
        self.mesh = mesh
        self.worst_ratio, self.max_abs = 0.0, 0.0
        self.worst_name = ""
        self.count = 0

    def check(self, actual, expected, name, rtol=5e-5, atol=5e-6):
        self.count += 1
        if actual is None or actual.shape != expected.shape:
            self.worst_ratio = float("inf"); self.worst_name = name + " missing/wrong shape"
            return
        x, y = actual.detach().double(), expected.detach().double()
        error = (x-y).abs()
        if not torch.isfinite(error).all():
            ratio, absolute = float("inf"), float("inf")
        else:
            absolute = error.max().item()
            ratio = (error/(atol+rtol*y.abs()).clamp_min(1e-300)).max().item()
        self.max_abs = max(self.max_abs, absolute)
        if ratio > self.worst_ratio:
            self.worst_ratio, self.worst_name = ratio, name

    def finish(self, case):
        result=torch.tensor([self.worst_ratio,self.max_abs],device=self.mesh.device,dtype=torch.float64)
        if dist.is_initialized(): dist.all_reduce(result,op=dist.ReduceOp.MAX)
        if self.worst_ratio > 1:
            print(f"rank={self.mesh.rank} worst={self.worst_name} tolerance_ratio={self.worst_ratio}",flush=True)
        if result[0].item() > 1:
            raise AssertionError(f"{case} failed; max normalized error={result[0].item()}")
        if self.mesh.rank == 0:
            print(f"PASS case={case} tp={self.mesh.dims.tp} pp={self.mesh.dims.pp} dp={self.mesh.dims.dp} "
                  f"max_abs_error={result[1].item():.3e} tolerance_ratio={result[0].item():.3e}",flush=True)


def random_tensor(shape, seed, device):
    return torch.randn(shape,generator=torch.Generator().manual_seed(seed),dtype=torch.float64).to(device)


def check_groups(mesh, verify):
    for axis in ("tp","pp","dp"):
        value=torch.tensor(float(mesh.rank+1),device=mesh.device)
        sum_(value,mesh,axis)
        members=mesh.members.get(axis,(mesh.rank,))
        expected=torch.tensor(float(sum(r+1 for r in members)),device=mesh.device)
        verify.check(value,expected,axis+" group SUM",rtol=0,atol=0)


def check_linear(mesh, verify):
    tp,t=mesh.dims.tp,mesh.tp_rank
    # Column: 各 rank 的局部 loss 是全局分支 loss 的一部分，dX 必须合并。
    x=random_tensor((2,3,8),1,mesh.device).requires_grad_()
    w=random_tensor((16,8),2,mesh.device).requires_grad_()
    bias=random_tensor((16,),3,mesh.device).requires_grad_()
    local_x=x.detach().clone().requires_grad_()
    col=ColumnParallelLinear(8,16,mesh).to(device=mesh.device,dtype=torch.float64)
    with torch.no_grad():
        col.weight.copy_(w.chunk(tp,0)[t]); col.bias.copy_(bias.chunk(tp,0)[t])
    ref=F.linear(x,w,bias); out=col(local_x)
    out.square().sum().backward(); ref.square().sum().backward()
    verify.check(out,ref.chunk(tp,-1)[t],"column output")
    verify.check(local_x.grad,x.grad,"column input grad")
    verify.check(col.weight.grad,w.grad.chunk(tp,0)[t],"column weight grad")
    verify.check(col.bias.grad,bias.grad.chunk(tp,0)[t],"column bias grad")
    # Row: replicated output/loss；backward identity，不重复求和。
    x=random_tensor((2,3,16),4,mesh.device).requires_grad_()
    w=random_tensor((8,16),5,mesh.device).requires_grad_()
    bias=random_tensor((8,),6,mesh.device).requires_grad_()
    local_x=x.detach().chunk(tp,-1)[t].clone().requires_grad_()
    row=RowParallelLinear(16,8,mesh).to(device=mesh.device,dtype=torch.float64)
    with torch.no_grad():
        row.weight.copy_(w.chunk(tp,1)[t]); row.bias.copy_(bias)
    ref=F.linear(x,w,bias); out=row(local_x)
    out.square().sum().backward(); ref.square().sum().backward()
    verify.check(out,ref,"row output (bias must not multiply by TP)")
    verify.check(local_x.grad,x.grad.chunk(tp,-1)[t],"row input grad")
    verify.check(row.weight.grad,w.grad.chunk(tp,1)[t],"row weight grad")
    verify.check(row.bias.grad,bias.grad,"row replicated bias grad")


def check_embedding(mesh, verify):
    weight=random_tensor((16,8),7,mesh.device).requires_grad_()
    tokens=torch.tensor([[0,7,8,15],[1,0,15,9]],device=mesh.device)
    layer=VocabParallelEmbedding(16,8,mesh).to(device=mesh.device,dtype=torch.float64)
    with torch.no_grad(): layer.weight.copy_(weight.chunk(mesh.dims.tp,0)[mesh.tp_rank])
    ref=F.embedding(tokens,weight); out=layer(tokens)
    ref.square().sum().backward(); out.square().sum().backward()
    verify.check(out,ref,"embedding output")
    verify.check(layer.weight.grad,weight.grad.chunk(mesh.dims.tp,0)[mesh.tp_rank],"embedding masked grad")


def check_loss(mesh, verify):
    targets=torch.tensor([[0,7,15],[8,1,14]],device=mesh.device)
    for scale in (1.0,800.0):
        full=(random_tensor((2,3,16),8,mesh.device)*scale).requires_grad_()
        local=full.detach().chunk(mesh.dims.tp,-1)[mesh.tp_rank].clone().requires_grad_()
        ref=F.cross_entropy(full.reshape(-1,16),targets.reshape(-1),reduction="none").reshape(2,3)
        out=vocab_parallel_cross_entropy(local,targets,mesh)
        upstream=random_tensor((2,3),9,mesh.device)
        out.backward(upstream); ref.backward(upstream)
        verify.check(out,ref,f"vocab CE scale={scale}",rtol=1e-8,atol=1e-8)
        verify.check(local.grad,full.grad.chunk(mesh.dims.tp,-1)[mesh.tp_rank],"vocab CE grad",rtol=1e-8,atol=1e-8)


def check_model(mesh, verify):
    cfg=ModelConfig(seq_len=8)
    dense=build_dense(cfg,mesh.device); stage=build_stage(cfg,mesh)
    batch=make_batch(cfg,2,4,device=mesh.device)
    ref=dense(batch.input_ids); out=stage(batch.input_ids)
    verify.check(out,ref.chunk(mesh.dims.tp,-1)[mesh.tp_rank],"model local logits")
    language_model_loss(out,batch.target_ids,mesh).backward()
    F.cross_entropy(ref.reshape(-1,cfg.vocab_size),batch.target_ids.reshape(-1)).backward()
    full=dict(dense.named_parameters())
    for name,p in stage.named_parameters():
        verify.check(p.grad,shard_like(full[name].grad,p,mesh),name+" grad")


def optimizer(model):
    # 略大的 eps 减少理论零梯度（如 key bias）舍入噪声被 Adam 放大的影响。
    return torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.01,eps=1e-6)


def check_step(mesh, verify, steps, microbatches, clip):
    cfg=ModelConfig(seq_len=8)
    dense=build_dense(cfg,mesh.device); stage=build_stage(cfg,mesh)
    ref_opt=optimizer(dense); opt=optimizer(stage)
    full=dict(dense.named_parameters())
    for step in range(steps):
        local_batch=2*microbatches
        global_batch=make_batch(cfg,step,local_batch*mesh.dims.dp,device=mesh.device)
        batch=make_batch(cfg,step,local_batch,mesh.dp_rank,mesh.dims.dp,mesh.device)
        ref_opt.zero_grad(set_to_none=True)
        logits=dense(global_batch.input_ids)
        ref_loss=F.cross_entropy(logits.reshape(-1,cfg.vocab_size),global_batch.target_ids.reshape(-1))
        ref_loss.backward()
        ref_norm=torch.nn.utils.clip_grad_norm_(dense.parameters(),clip) if clip>0 else None
        ref_opt.step()
        actual=train_step(stage,opt,split_microbatches(batch,2),clip)
        verify.check(actual["loss"],ref_loss,f"step {step} global mean loss")
        if clip>0: verify.check(actual["grad_norm"],ref_norm,f"step {step} global grad norm")
        for name,p in stage.named_parameters():
            verify.check(p.grad,shard_like(full[name].grad,p,mesh),f"step {step} {name} grad")
            verify.check(p,shard_like(full[name],p,mesh),f"step {step} {name} updated param",atol=2e-5)


def check_checkpoint(mesh, verify, microbatches):
    cfg=ModelConfig(seq_len=8)
    a=build_stage(cfg,mesh); opt=optimizer(a)
    def batches(step):
        return split_microbatches(make_batch(cfg,step,2*microbatches,mesh.dp_rank,mesh.dims.dp,mesh.device),2)
    train_step(a,opt,batches(0))
    # 每个 rank 的本地临时目录足以测同 rank shard 的保存和恢复，不依赖共享文件系统。
    with tempfile.TemporaryDirectory(prefix=f"mini-check-r{mesh.rank}-") as directory:
        save_checkpoint(directory,a,opt,1)
        expected_rng=torch.rand(5)
        expected_cuda_rng=torch.rand(5,device=mesh.device) if mesh.device.type=="cuda" else None
        expected=train_step(a,opt,batches(1))
        b=build_stage(cfg,mesh); restored=optimizer(b)
        next_step=load_checkpoint(directory,b,restored)
        verify.check(torch.tensor(next_step),torch.tensor(1),"checkpoint data cursor",rtol=0,atol=0)
        verify.check(torch.rand(5),expected_rng,"CPU RNG resume",rtol=0,atol=0)
        if expected_cuda_rng is not None:
            verify.check(torch.rand(5,device=mesh.device),expected_cuda_rng,"CUDA RNG resume",rtol=0,atol=0)
        # 固定测试 batch，错误的 next_step 由上面的独立检查捕获，不破坏 collective 顺序。
        actual=train_step(b,restored,batches(1))
        verify.check(actual["loss"],expected["loss"],"resumed next loss",rtol=0,atol=1e-7)
        for (name,p),q in zip(a.named_parameters(),b.parameters()):
            verify.check(q,p,name+" resumed next update",rtol=0,atol=1e-7)
            for key in ("step","exp_avg","exp_avg_sq"):
                verify.check(restored.state[q][key],opt.state[p][key],name+" "+key,rtol=0,atol=1e-7)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=["groups","linear","embedding","loss","model","step","checkpoint"],required=True)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    for axis in ("tp","pp","dp"): parser.add_argument("--"+axis,type=int,default=1)
    parser.add_argument("--microbatches",type=int,default=3)
    parser.add_argument("--steps",type=int,default=3)
    parser.add_argument("--clip",type=float,default=0.0)
    args=parser.parse_args()
    dims=ParallelDims(args.tp,args.pp,args.dp)
    ModelConfig(seq_len=8).validate(dims.tp,dims.pp)
    if args.microbatches<1 or args.steps<1 or args.clip<0: parser.error("invalid steps/microbatches/clip")
    if args.case in ("linear","embedding","loss","model") and (dims.pp!=1 or dims.dp!=1):
        parser.error("this isolated case requires PP=DP=1; use --case step for mixed parallelism")
    required={"groups":[],"linear":["T03","T04","T05","T06"],
              "embedding":["T04","T07"],"loss":["T08","T09"],
              "model":model_tasks()+(["T01"] if dims.tp==1 else ["T08","T09"])}
    ids=required.get(args.case,training_tasks(dims,clip=args.clip>0,
                     save=args.case=="checkpoint",resume=args.case=="checkpoint"))
    if dims.world_size>1: ids=ids+["T02"]
    require_tasks(ids)
    torch.set_num_threads(1)
    mesh=initialize(dims,args.device)
    try:
        verify=Verifier(mesh)
        if args.case=="step": check_step(mesh,verify,args.steps,args.microbatches,args.clip)
        elif args.case=="checkpoint": check_checkpoint(mesh,verify,args.microbatches)
        else:
            {"groups":check_groups,"linear":check_linear,"embedding":check_embedding,
             "loss":check_loss,"model":check_model}[args.case](mesh,verify)
        verify.finish(args.case)
    finally:
        shutdown()


if __name__=="__main__":
    main()
