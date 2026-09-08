"""Snapshot preparation; reuse T50 for all-rank manifest/atomic file publication."""
from megatron_mini.exercise import todo

def snapshot_offloaded(optimizers,transfers,metadata,*,next_step):
    """T72: wait all Transfer.wait_host() and clone CPU-only state into an immutable
    value snapshot (no storage alias to live optimizer). Return dict with exactly:
    format_version=1, metadata (deep copied), next_step, owners (list of state_dict()).
    Reject negative next_step. Snapshot after globally agreed optimizer update/skip,
    not mid-DMA/mid-microbatch. CPUShardAdam.state_dict already clones current data.
    Caller supplies model shard IDs, offsets, topology, precision/offload policy,
    scaler, RNG and data cursor in metadata. Transient full buffers and SAC caches
    are NOT checkpoint state. T50 commits files; T60 handles supported offline reshard.
    Restoring on a new placement means rebuilding transfer streams/events, not loading
    stale CUDA addresses. Test snapshot isolation and next optimizer update after load.
    """
    todo('T72')
