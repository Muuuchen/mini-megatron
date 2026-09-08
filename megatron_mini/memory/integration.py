"""Additional composition task; it does not hide missing mechanisms behind a fake flag."""
from megatron_mini.exercise import todo

def build_memory_runtime(config,model_config,device):
    """T71: return TrainRuntime for zero3_cpu / zero3_cpu_sac / fsdp_cpu_sac recipes.
    T55 owns mesh/model construction; factor a shared assembly helper to avoid recursion.
    Compose T62 unit boundaries, T64 CPU canonical shards, T65 transfers, T63 CPU Adam,
    and T69/T70 selective replay. A custom autograd Function receiving a CPU local-shard
    input must return its dlocal on CPU; NCCL AG/RS stay on device staging tensors.
    CPUParamStore and optimizer must share ONE authoritative shard, not copied shadows.
    Accumulate all microbatch gradients into correct owner before global finite-check;
    convert found_inf into device flags for NCCL, agree globally, clip final shards,
    CPU update only when finite, then invalidate stale staging/full buffers.
    SAC wraps pure unit computation only, not parameter fetch / token dispatch / RS.
    Activation offload can target checkpoint inputs via explicit T66 records; generic
    saved_tensors_hooks do NOT automatically offload SAC's internal Python op-cache.
    Baseline SAC outputs remain device-resident; account for them in the memory budget.
    All offload/comm handles drain before checkpoint T72/T50 and before safe pool reuse.
    Required final comparison: same data, M=2, 3 AdamW updates, moments, RNG, clipping,
    one overflow skip and resume-next-update versus nonoffloaded reference.
    CPU local cases do not validate actual H2D/D2H or this whole recipe. Explicitly
    reject unsupported PP/EP/precision combinations rather than silently falling back.
    """
    todo('T71')
