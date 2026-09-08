"""入口：baseline 已完成；student 在相关 TODO 完成后启用。"""
import argparse
import json
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn import functional as F
from .config import ModelConfig
from .data import make_batch, split_microbatches
from .distributed.topology import ParallelDims
from .distributed.mesh import initialize, shutdown
from .model.dense import build_dense
from .model.state import build_stage
from .training.engine import train_step
from .checkpoint import save_checkpoint, load_checkpoint
from .tasks import require_tasks, training_tasks


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["baseline", "student"], default="baseline")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    for axis in ("tp", "pp", "dp"):
        p.add_argument("--" + axis, type=int, default=1)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--microbatches", type=int, default=4)
    p.add_argument("--steps", type=int, default=20, help="total global steps, not extra resume steps")
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--max-grad-norm", type=float, default=0.0, help="0 disables clipping")
    p.add_argument("--checkpoint-dir", type=Path, default=None, help="save final same-topology state")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None, help="new JSONL metrics file (will overwrite)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(1)
    cfg = ModelConfig(**json.loads(args.config.read_text())) if args.config else ModelConfig()
    dims = ParallelDims(args.tp, args.pp, args.dp)
    cfg.validate(dims.tp, dims.pp)
    if min(args.micro_batch, args.microbatches, args.steps) < 1 or args.lr <= 0:
        raise SystemExit("batch sizes, steps, and learning rate must be positive")
    if args.max_grad_norm < 0:
        raise SystemExit("max-grad-norm must be nonnegative")
    if args.mode == "baseline":
        if dims.world_size != 1 or args.resume or args.checkpoint_dir:
            raise SystemExit("baseline is single-process only, without checkpoint exercises")
    else:
        # 在 init_process_group 之前统一检查，避免部分 rank 跑进 collective 才遇到 TODO。
        require_tasks(training_tasks(dims, args.max_grad_norm > 0,
                                     args.checkpoint_dir is not None, args.resume is not None))
    mesh = initialize(dims, args.device)
    try:
        model = build_dense(cfg, mesh.device) if args.mode == "baseline" else build_stage(cfg, mesh)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        if args.mode == "student":
            model.run_config = {"micro_batch": args.micro_batch, "microbatches": args.microbatches,
                                "lr": args.lr, "max_grad_norm": args.max_grad_norm}
        start = load_checkpoint(args.resume, model, optimizer) if args.resume else 0
        if start >= args.steps:
            raise ValueError(f"resume next_step={start} must be less than --steps={args.steps}")
        if args.output and mesh.rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("")
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats(mesh.device)
        global_tokens = args.micro_batch * args.microbatches * dims.dp * cfg.seq_len
        times = []
        for step in range(start, args.steps):
            batch = make_batch(cfg, step, args.micro_batch * args.microbatches,
                               mesh.dp_rank, dims.dp, mesh.device)
            micro = split_microbatches(batch, args.micro_batch)
            if dist.is_initialized(): dist.barrier()
            if args.device == "cuda": torch.cuda.synchronize(mesh.device)
            t0 = time.perf_counter()
            if args.mode == "baseline":
                optimizer.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device=mesh.device)
                for mb in micro:
                    logits = model(mb.input_ids)
                    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), mb.target_ids.reshape(-1))
                    loss_sum += loss.detach() / len(micro)
                    (loss / len(micro)).backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                result = {"loss": loss_sum}
            else:
                result = train_step(model, optimizer, micro, args.max_grad_norm)
            if args.device == "cuda": torch.cuda.synchronize(mesh.device)
            elapsed = torch.tensor(time.perf_counter()-t0, device=mesh.device, dtype=torch.float64)
            if dist.is_initialized(): dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            ms = elapsed.item()*1000
            peak = torch.tensor(torch.cuda.max_memory_allocated(mesh.device) if args.device=="cuda" else 0,
                                device=mesh.device, dtype=torch.float64)
            if dist.is_initialized(): dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            row = {"step": step, "loss": result["loss"].item(), "step_ms": ms,
                   "global_tokens_per_second": global_tokens/elapsed.item(),
                   "max_rank_peak_allocated_bytes": int(peak.item()),
                   "mode": args.mode, "tp": dims.tp, "pp": dims.pp, "dp": dims.dp,
                   "global_batch": args.micro_batch*args.microbatches*dims.dp,
                   "warmup": step < start+2}
            if step >= start+2: times.append(ms)
            if mesh.rank == 0:
                print(json.dumps(row), flush=True)
                if args.output:
                    with args.output.open("a") as f: f.write(json.dumps(row)+"\n")
        if args.checkpoint_dir:
            save_checkpoint(args.checkpoint_dir, model, optimizer, args.steps)
        if mesh.rank == 0 and times:
            print(f"steady_state_mean_step_ms={sum(times)/len(times):.3f} (first 2 steps excluded)")
    finally:
        shutdown()


if __name__ == "__main__":
    main()
