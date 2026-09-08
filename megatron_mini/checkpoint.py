"""仅同拓扑 resume：每个 global rank 保存一个文件，包含 DP 副本（教学冗余）。"""
from dataclasses import asdict
from pathlib import Path
import os
import torch
from megatron_mini.exercise import todo


def rank_file(directory, mesh):
    return Path(directory) / f"rank_{mesh.rank:04d}.pt"


def metadata(stage):
    return {"format_version": 1, "dims": asdict(stage.mesh.dims),
            "rank": stage.mesh.rank, "config": asdict(stage.cfg),
            "device_type": stage.mesh.device.type, "run_config": getattr(stage, "run_config", None)}


def validate_metadata(saved, stage):
    if saved != metadata(stage):
        raise ValueError("checkpoint metadata mismatch: only same config/topology/rank is supported")


def save_checkpoint(directory, stage, optimizer, next_step: int):
    """T19: 只在 train_step 完成后调用；恢复从 next_step 开始，不重放当前 step。

    保存 metadata(stage)、stage.state_dict()、optimizer.state_dict()、next_step，
    以及 torch CPU RNG（torch.get_rng_state）和当前设备 CUDA RNG（或 None）。
    文件 key 统一为 meta/model/optimizer/next_step/cpu_rng/cuda_rng。
    make_batch 是 step-indexed，因此 next_step 就是本 lab 数据游标。
    mkdir 后先写 rank 独享临时文件，再 os.replace 到 rank_file；避免半写单文件。
    所有 rank 都调用。没有跨 rank 事务；中断后的部分目录不保证可恢复。
    """
    if next_step < 0:
        raise ValueError("next_step must be nonnegative")
    todo("T19")


def load_checkpoint(directory, stage, optimizer) -> int:
    """T20: 先验证 metadata，再恢复模型、optimizer、RNG，返回 next_step。

    使用 torch.load(...,map_location='cpu',weights_only=True)；state 使用 tensor、
    Python 基础类型，适用于本 lab。仅加载自己产生的可信 checkpoint。
    CPU RNG tensor 保持 CPU；model/optimizer 通过 load_state_dict 迁移至参数设备。
    CUDA 时用 torch.cuda.set_rng_state 恢复当前设备状态。
    不支持换 TP/PP/DP、reshape 模型或通用 Megatron/HF checkpoint 转换。
    """
    todo("T20")
