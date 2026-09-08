"""完整 orchestration；学生补调度/梯度语义，框架负责调用顺序。"""
from megatron_mini.distributed.mesh import reported_loss
from megatron_mini.pipeline_parallel.schedules import gpipe_forward, gpipe_backward
from .schedules import forward_backward_no_pipeline
from .grad import average_dp_gradients, clip_grad_norm


def train_step(stage, optimizer, microbatches, max_grad_norm=0.0):
    optimizer.zero_grad(set_to_none=True)
    if stage.mesh.dims.pp == 1:
        local_loss = forward_backward_no_pipeline(stage, microbatches)
    else:
        records = gpipe_forward(stage, microbatches)
        local_loss = gpipe_backward(records, stage.mesh)
    average_dp_gradients(stage, stage.mesh)
    norm = None
    if max_grad_norm > 0:
        norm = clip_grad_norm(stage, stage.mesh, max_grad_norm)
    optimizer.step()  # 每一个 local parameter、每一个 global step 恰好更新一次。
    return {"loss": reported_loss(local_loss, stage.mesh), "grad_norm": norm}
